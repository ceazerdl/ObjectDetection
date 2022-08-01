import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
import torch
import numpy as np
import torchvision.transforms as transform
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.models as models
import dataset
from matplotlib import pyplot as plt


class CFG(object):
    seed = 111
    Base_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path_state_dict = os.path.join(Base_dir, "data", "alexnet-owt-4df8aa71.pth")
    data_dir = os.path.join(Base_dir, "data", "train")
    ckpt_pathname = "alexnet_img224224_bs32.pth"
    img_size = [224, 224]
    train_bs = 32
    valid_bs = 2 * train_bs
    num_classes = 2
    epoch = 90
    lr = 1e-2
    wd = 5e-4
    thr = 0.5
    lr_drop = 10


def build_model(CFG, vis_model=False):
    model = models.alexnet(pretrained=False)
    pretrain_weights = torch.load(CFG.path_state_dict)
    model.load_state_dict(pretrain_weights)

    if vis_model:
        from torchsummary import summary
        summary(model, input_size=(3, 224, 224), device="cpu")
    model.to(CFG.device)
    return model


