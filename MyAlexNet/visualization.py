import torch
import os
import torch.nn as nn
import torchvision.transforms as transform
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import torchvision.models as models
from PIL import Image

Base_dir = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    path = "./results"
    if not os.path.exists(path):
        os.makedirs(path)
    log_dir = os.path.join(Base_dir, "results")

    writer = SummaryWriter(log_dir, filename_suffix="_kernel")
    path_state_dict = os.path.join(Base_dir, "data", "alexnet-owt-4df8aa71.pth")
    model = models.alexnet()
    model.load_state_dict(torch.load(path_state_dict))

    # ----------------------------卷积核可视化-------------------------------
    kernel_num = -1
    vis_max = 1
    for sub_module in model.modules():
        if not isinstance(sub_module, nn.Conv2d):
            continue
        kernel_num += 1
        if kernel_num > vis_max:
            break
        kernels = sub_module.weight
        print("kernels.size: " % tuple(kernels.shape))
        c_out, c_in, k_h, k_w = tuple(kernels.shape)

        # 拆分channels
        for idx in range(c_out):
            # 获得(3, h, w)，但是make_grid函数需要的是bchw格式的tensor，很明显，weight是逐通道的，所以，改成(3, 1, h, w)，3个1维的weight
            kernel_idx = kernels[idx, ...].unsqueeze(1)
            kernel_grid = vutils.make_grid(kernel_idx, normalize=True, scale_each=True, nrow=c_in)
            print("kernel_grid.size(): " % tuple(kernel_grid.shape), end=" ")
            print()
            writer.add_image("{}_Convlayer_split_in_channel".format(kernel_num), kernel_grid, global_step=idx)
        kernel_all = kernels.view(-1, 3, k_h, k_w)
        kernel_grid = vutils.make_grid(kernel_all, normalize=True, scale_each=True)
        print("kernel_grid.size(): " % tuple(kernel_grid.shape), end=" ")
        print()
        writer.add_image("{}_all".format(kernel_num), kernel_grid, global_step=50)
        print("{}_convlayer shape:{}".format(kernel_num, tuple(kernels.shape)))
    writer.close()
    # ------------------------features visualization------------------------------------
    writer = SummaryWriter(log_dir=log_dir, filename_suffix="_feature map")

    # 数据
    path_img = os.path.join(Base_dir, "data", "tiger cat.jpg")  # your path to image
    normMean = [0.49139968, 0.48215827, 0.44653124]
    normStd = [0.24703233, 0.24348505, 0.26158768]
    img_transforms = transform.Compose([
        transform.Resize((224, 224)),
        transform.ToTensor(),
        transform.Normalize(normMean, normStd)
    ])

    img_pil = Image.open(path_img).convert('RGB')
    img_tensor = img_transforms(img_pil)
    img_tensor.unsqueeze_(0)  # chw --> bchw

    # forward提取第一层卷积后的特征图，可以使用钩子函数
    convlayer1 = model.features[0]
    fmap_1 = convlayer1(img_tensor)

    # 预处理，1张图提取了64张特征图，为了显示64张图，改变数据特征维度，变成64张1维的灰度图
    fmap_1.transpose_(0, 1)  # bchw=(1, 64, 55, 55) --> (64, 1, 55, 55)
    # Make a grid of images.
    fmap_1_grid = vutils.make_grid(fmap_1, normalize=True, scale_each=True, nrow=8)

    writer.add_image('feature map in conv1', fmap_1_grid, global_step=60)
    writer.close()













