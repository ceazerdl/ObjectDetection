import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt
from dataset import NCFMDateSet
from common_tools import get_googlenet
from tqdm import tqdm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    data_dir = os.path.join(BASE_DIR, "NCFM-train", "train")
    path_state_dict = os.path.join(BASE_DIR, "googlenet-1378be20.pth")

    num_classes = 8
    max_epoch = 7
    bs = 64
    LR = 1e-3
    log_interval = 1
    val_interval = 1
    start_epoch = -1
    lr_decay_step = 1

    # ================= step 1/5 数据处理 ========================
    norm_mean = [0.485, 0.496, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    normalizes = transforms.Normalize(norm_mean, norm_std)
    valid_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.TenCrop(224, vertical_flip=False),
        transforms.Lambda(lambda crops: torch.stack([normalizes(transforms.ToTensor()(crop)) for crop in crops]))
    ])

    # 构建Dataset实例
    train_data = NCFMDateSet(data_dir, mode="train", transform=train_transform)
    valid_data = NCFMDateSet(data_dir, mode="valid", transform=valid_transform)

    # 构建DataLoader实例
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_data, batch_size=bs, shuffle=False, pin_memory=True)

    # ============================= 2/5 模型 ===============================
    model = get_googlenet(path_state_dict, device, num_classes=8, vis_model=True)

    # ============================= 3/5 损失函数 ============================
    criterion = nn.CrossEntropyLoss()
    # ============================= 4/5 优化器 ==============================
    # 是否finetune
    flag = 0
    if flag:
        fc_params_id = list(map(id, model.fc.parameters()))
        base_params = filter(lambda p: id(p) not in fc_params_id, model.parameters())
        optimizer = optim.SGD([
            {"params": base_params, "lr": LR * 0.1},
            {"params": model.fc.parameters()}], lr=LR, momentum=0.9)
    else:
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=0.1)

    # ===================== setp 5/5 训练 ====================================
    for epoch in range(max_epoch):
        loss_mean = 0.
        correct = 0.
        total = 0.

        model.train()
        for i, (imgs, lbs) in tqdm(enumerate(train_loader)):
            inputs, labels = imgs.to(device), lbs.to(device)
            optimizer.zero_grad()
            # outputs为namedtuple
            outputs = model(inputs)
            loss_main, loss_aux1, loss_aux2 = criterion(outputs[0], labels), \
                                  criterion(outputs[1], labels), criterion(outputs[2], labels)
            loss = loss_main + 0.3 * loss_aux1 + 0.3 * loss_aux2
            loss.backward()

            optimizer.step()

            # 统计分类情况
            _, predicted = torch.max(outputs.logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).squeeze().cpu().sum().numpy()

            # 打印训练信息
            loss_mean += loss_main.item()
            if (i + 1) % log_interval == 0:
                loss_mean = loss_mean / log_interval
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss_main:{:.4f} Acc:{:.2%} lr:{}".format(
                    epoch, max_epoch, i + 1, len(train_loader), loss_mean, correct / total, scheduler.get_last_lr()))
                loss_mean = 0

        scheduler.step()

        # validate the model
        if (epoch + 1) % val_interval == 0:

            correct_val = 0.
            total_val = 0.
            loss_val = 0.
            model.eval()
            with torch.no_grad():
                for j, data in enumerate(valid_loader):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    bs, ncrops, c, h, w = inputs.size()
                    # outputs与self.training=True不同，此时不是namedtuple，是正常的结果(640*8)
                    outputs = model(inputs.view(-1, c, h, w))
                    outputs_avg = outputs.view(bs, ncrops, -1).mean(1)
                    loss = criterion(outputs_avg, labels)

                    _, predicted = torch.max(outputs_avg.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).squeeze().cpu().sum().numpy()

                    loss_val += loss.item()

                loss_val_mean = loss_val / len(valid_loader)

                print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, max_epoch, j + 1, len(valid_loader), loss_val_mean, correct_val / total_val))
            model.train()























