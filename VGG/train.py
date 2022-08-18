import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt
from my_dataset import CatDogDataset
from tqdm import tqdm
from common_tools import get_vgg16
import time


def build_transforms(CFG):
    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize(CFG.img_size[0]),
            transforms.CenterCrop(CFG.img_size[0]),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(CFG.norm_mean, CFG.norm_std)
        ]),
        "valid": transforms.Compose([
            transforms.Resize((CFG.img_size)),
            transforms.TenCrop(224, vertical_flip=False),      # 10张图片的列表
            # 将10张图片堆叠起来
            transforms.Lambda(lambda crops: torch.stack([CFG.nl(transforms.ToTensor()(crop)) for crop in crops]))
        ])
    }
    return data_transforms


def build_dataloader(CFG, data_transforms):
    train_data = CatDogDataset(data_dir=CFG.data_dir, mode="train", rng_seed=CFG.seed, transform=data_transforms["train"])
    valid_data = CatDogDataset(data_dir=CFG.data_dir, mode="valid", rng_seed=CFG.seed, transform=data_transforms["valid"])

    train_loader = DataLoader(train_data, batch_size=CFG.train_bs, num_workers=0, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_data, batch_size=CFG.valid_bs, num_workers=0, shuffle=False, pin_memory=True)

    return train_loader, valid_loader


class CFG(object):
    seed = 111,
    Base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(Base_dir, "data", "train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path_state_dict = os.path.join(Base_dir, "data", "vgg16-397923af.pth")
    ckpt_pathname = "vgg16_img224224_bs64.pth"
    img_size = [256, 256]
    train_bs = 64    # 原文256
    valid_bs = train_bs * 2
    num_classes = 2
    epoch = 3   # 原文没有预训练模型，epoch=74
    lr = 1e-5   # 原文lr = 1e-2
    wd = 5e-4
    lr_drop = 10
    log_interval = 5
    is_best = False
    best_val_acc = 0
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    nl = transforms.Normalize(norm_mean, norm_std)


def build_model(CFG, vis_model=False):
    model = get_vgg16(CFG.path_state_dict, CFG.device, vis_model)
    return model


def train_one_epoch(model, train_loader, optimizer, lossfunc, CFG, total, correct, loss_mean, epoch):
    model.train()
    for i, (images, labels) in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()

        # forward
        images, labels = images.to(CFG.device), labels.to(CFG.device)
        outputs = model(images)

        # backward
        loss = lossfunc(outputs, labels)
        loss.backward()

        # updata weights
        optimizer.step()

        # 统计分类情况
        _, predicted = torch.max(outputs.data, 1)   # outputs.shape为(bs, numclass)       predicted.shape为(bs, 1)
        total += labels.size(0)
        correct += (predicted == labels).squeeze().cpu().sum().numpy()

        # 打印训练信息
        loss_mean += loss.item()
        if (i + 1) % CFG.log_interval == 0:
            loss_mean = loss_mean / CFG.log_interval
            print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss:{:.4f} Acc:{:.2%}".format(
                epoch, CFG.epoch - 1, i + 1, len(train_loader), loss_mean, correct / total
            ))

            loss_mean = 0


@torch.no_grad()
def valid_one_epoch(model, valid_loader, CFG, correct_val, total_val, loss_val, losses, epoch):
    model.eval()
    for i, (images, labels) in tqdm(enumerate(valid_loader)):
        images, labels = images.to(CFG.device), labels.to(CFG.device)
        bs, ncrops, c, h, w = images.size()
        outputs = model(images.view(-1, c, h, w))
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)
        loss = losses(outputs_avg, labels)

        _, predicted = torch.max(outputs_avg.data, 1)
        total_val += labels.size(0)
        correct_val += (predicted == labels).squeeze().cpu().sum().numpy()

        loss_val += loss.item()

    loss_val_mean = loss_val / len(valid_loader)
    # valid_curve.append(loss_val_mean)

    print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
        epoch, CFG.epoch - 1, i + 1, len(valid_loader), loss_val_mean, correct_val / total_val
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
    flag = 1
    if flag:
        fc_params_id = list(map(id, model.classifier.parameters()))
        base_params = filter(lambda p: id(p) not in fc_params_id, model.parameters())
        optimizer = optim.SGD([{"params": base_params, "lr": CFG.lr * 0.1},  # 如果为0，则不更新卷积层参数
                              {"params": model.classifier.parameters(), "lr": CFG.lr}], momentum=0.9, weight_decay=CFG.wd)
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



































