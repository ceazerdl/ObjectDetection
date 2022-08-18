import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
import torch
import numpy as np
import torchvision.transforms as transform
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.models as models
from dataset import CatAndDogDataset
from matplotlib import pyplot as plt
from tqdm import tqdm
import time


def build_transforms(CFG):
    data_transforms = {
        "train": transform.Compose([
            # 注意256和(256, 256)的区别
            # size (sequence or int): Desired output size. If size is a sequence like
            # (h, w), the output size will be matched to this. If size is an int,
            # the smaller edge of the image will be matched to this number maintaining
            # the aspect ratio.
            transform.Resize((CFG.img_size[0])),
            transform.CenterCrop(CFG.img_size[0]),
            transform.RandomCrop(224),
            transform.RandomHorizontalFlip(p=0.5),
            transform.ToTensor(),
            transform.Normalize(CFG.norm_mean, CFG.norm_std)
        ]),
        "valid": transform.Compose([
            transform.Resize((CFG.img_size)),
            transform.TenCrop(224, vertical_flip=False),
            transform.Lambda(lambda crops: torch.stack([CFG.nl(transform.ToTensor()(crop)) for crop in crops]))
        ])
    }
    return data_transforms


def build_dataloader(CFG, data_transforms):
    train_data = CatAndDogDataset(data_dir=CFG.data_dir, mode="train", rng_seed=CFG.seed, transform=data_transforms["train"])
    valid_data = CatAndDogDataset(data_dir=CFG.data_dir, mode="valid", rng_seed=CFG.seed, transform=data_transforms["valid"])

    train_dataloader = DataLoader(train_data, batch_size=CFG.train_bs, num_workers=4, shuffle=True, pin_memory=True)
    valid_dataloader = DataLoader(valid_data, batch_size=CFG.valid_bs, num_workers=4, shuffle=False, pin_memory=True)

    return train_dataloader, valid_dataloader


class CFG(object):
    seed = 111
    Base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(Base_dir, "data", "train")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path_state_dict = os.path.join(Base_dir, "data", "alexnet-owt-4df8aa71.pth")
    ckpt_pathname = "alexnet_img224224_bs32.pth"
    img_size = [256, 256]
    train_bs = 64
    valid_bs = 2 * train_bs
    num_classes = 2
    epoch = 15
    lr = 1e-5
    wd = 5e-4
    lr_drop = 10
    log_interval = 5
    is_best = False
    best_val_acc = 0
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    nl = transform.Normalize(norm_mean, norm_std)


def build_model(CFG, vis_model=False):
    model = models.alexnet(pretrained=False)
    pretrain_weights = torch.load(CFG.path_state_dict)
    model.load_state_dict(pretrain_weights)

    if vis_model:
        from torchsummary import summary
        summary(model, input_size=(3, 224, 224), device="cpu")

    # 更改model的输出层
    num_features = model.classifier._modules['6'].in_features
    model.classifier._modules['6'] = nn.Linear(num_features, CFG.num_classes)
    model.to(CFG.device)
    return model


def train_one_epoch(model, train_loader, optimizer, lossfunc, CFG, total, correct, loss_mean, epoch):
    model.train()
    for i, (images, labels) in tqdm(enumerate(train_loader)):

        optimizer.zero_grad()

        # forward
        images, labels = images.to(CFG.device), labels.to(CFG.device)
        output = model(images)

        # backward
        loss = lossfunc(output, labels)
        loss.backward()

        # update weights
        optimizer.step()

        # 统计分类情况
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).squeeze().cpu().sum().numpy()

        # 打印训练信息
        loss_mean += loss.item()
        # train_curve.append(loss.item())
        if (i+1) % CFG.log_interval == 0:
            loss_mean = loss_mean / CFG.log_interval
            print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss:{:.4f} Acc:{:.2%}".format(
                epoch, CFG.epoch-1, i+1, len(train_loader), loss_mean, correct/total
            ))

            loss_mean = 0.


@torch.no_grad()
def valid_one_epoch(model, valid_loader, CFG, correct_val, total_val, loss_val, losses, epoch):
    model.eval()
    for i, (images, labels) in tqdm(enumerate(valid_loader)):
        images, labels = images.to(CFG.device), labels.to(CFG.device)  # b, ncrops, c, h, w
        bs, ncrops, c, h, w = images.size()
        outputs = model(images.view(-1, c, h, w))
        output_avg = outputs.view(bs, ncrops, -1).mean(1)

        loss = losses(output_avg, labels)

        _, predicted = torch.max(output_avg.data, 1)
        total_val += labels.size(0)
        correct_val += (predicted == labels).squeeze().cpu().sum().numpy()

        loss_val += loss.item()

    loss_val_mean = loss_val / len(valid_loader)
    # valid_curve.append(loss_val_mean)

    print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
        epoch, CFG.epoch-1, i+1, len(valid_loader), loss_val_mean, correct_val / total_val
    ))
    return correct_val / total_val


if __name__ == "__main__":
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name())
    ckpt_path = f"./checkpoint"
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    datatransform = build_transforms(CFG)
    train_loader, valid_loader = build_dataloader(CFG, datatransform)
    model = build_model(CFG)
    criterion = nn.CrossEntropyLoss()

    # finetune
    flag = 0
    if flag:
        fc_params_id = list(map(id, model.classifier.parameters()))
        base_params = filter(lambda p: id(p) not in fc_params_id, model.parameters())
        optimizer = optim.SGD([{"params":base_params, "lr": CFG.lr * 0.1},  # 如果为0，则不更新卷积层参数
                              {"params":fc_params_id, "lr": CFG.lr}], momentum=0.9, weight_decay=CFG.wd)
    else:
        optimizer = optim.SGD(model.parameters(), lr=CFG.lr, momentum=0.9, weight_decay=CFG.wd)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, CFG.lr_drop, gamma=0.1)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(patience=5)  # lr_scheduler.step(val_loss)更新的时候给个监控的值

    for epoch in range(CFG.epoch):
        start_time = time.time()
        loss_mean = 0
        correct = 0
        total = 0

        train_one_epoch(model, train_loader, optimizer, criterion, CFG, total, correct, loss_mean, epoch)

        lr_scheduler.step()

        correct_val = 0
        total_val = 0
        loss_val = 0
        val_acc = valid_one_epoch(model, valid_loader, CFG, correct_val, total_val, loss_val, criterion, epoch)
        CFG.is_best = (val_acc > CFG.best_val_acc)
        if CFG.is_best:
            CFG.best_val_acc = val_acc
            save_path = f"{ckpt_path}/{CFG.best_val_acc}{CFG.ckpt_pathname}"
            if os.path.isfile(save_path):
                os.remove(save_path)
            torch.save(model.state_dict(), save_path)

        epoch_time = time.time() - start_time
        print("epoch:{}, time:{:.2f}s, best:{}\n".format(epoch, epoch_time, CFG.best_val_acc), flush=True)





















































