import os
os.environ["NLS_LANG"] = "SIMPLIFIED CHINESE_CHINA.UTF8"
import time
import json
import torch
import torch.nn as nn
import torchvision.transforms as transform
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.models as models


Base_dir = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_label = ["cat", "dog"]


def build_transform(img, transform=None):
    """
    :param img: PIL格式，因为torchvision.transforms接收的是PIL格式的图片
    :param transform:
    :return:
    """
    if transform is None:
        raise Exception("imgs should be transformed")
    img = transform(img)
    return img


def load_class_names(p_clsnames, p_clsnames_cn):
    """
    加载标签名
    :param p_clsnames:
    :param p_clsnames_cn:
    :return:
    """
    with open(p_clsnames, "r") as f:
        class_names = json.load(f)
    with open(p_clsnames_cn, encoding="UTF-8") as f:
        class_names_cn = f.readlines()

    return class_names, class_names_cn


def get_model(path_state_dict, vis_model=False):
    """
    创建模型，加载参数
    :param path_state_dict:
    :param vis_model:
    :return:
    """
    model = models.alexnet()

    # 更改model的输出层
    num_features = model.classifier._modules['6'].in_features
    model.classifier._modules['6'] = nn.Linear(num_features, 2)

    model.load_state_dict(torch.load(path_state_dict))

    model.eval()

    if vis_model:
        from torchsummary import summary
        summary(model, input_size=(3, 224, 224), device="cpu")

    model.to(device)

    return model


def process_img(path_img):
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    infer_transform = transform.Compose([
        transform.Resize(256),
        transform.CenterCrop((224, 224)),
        transform.ToTensor(),
        transform.Normalize(norm_mean, norm_std)
    ])

    img = Image.open(path_img).convert("RGB")

    img_tensor = build_transform(img, infer_transform)
    # c,h,w -> b,c,h,w
    img_tensor.unsqueeze_(0)

    img_tensor = img_tensor.to(device)
    return img_tensor, img


if __name__ == "__main__":
    path_state_dict = os.path.join(Base_dir, "checkpoint", "0.9556alexnet_img224224_bs32.pth")
    path_img = os.path.join(Base_dir, "data", "tiger cat.jpg")
    # path_classnames = os.path.join(Base_dir, "data", "imagenet1000.json")
    # path_classnames_cn = os.path.join(Base_dir, "data", "imagenet_classnames.txt")

    # cls_n, cls_n_cn = load_class_names(path_classnames, path_classnames_cn)

    img_tensor, img = process_img(path_img)
    model = get_model(path_state_dict, True)

    with torch.no_grad():
        start_time = time.time()
        outputs = model(img_tensor)
        cost_time = time.time() - start_time

    _, pred_int = torch.max(outputs.data, 1)
    # _, top5_idx = torch.topk(outputs.data, 5, dim=1)

    pred_idx = int(pred_int.cpu().numpy())
    # pred_str, pred_cn = cls_n[pred_idx], cls_n_cn[pred_idx]
    # os.path.basename(path) 返回文件名
    print("img: {} is: {}\n".format(os.path.basename(path_img), data_label[pred_idx]))
    print("time consuming:{:.2f}s".format(cost_time))

    # visualization
    plt.imshow(img)
    plt.title(f"predict:{data_label[pred_idx]}")
    plt.text(5, 15+pred_idx*30, "top {}:{}".format(1, data_label[pred_idx]), bbox=dict(fc="yellow"))
    plt.show()



























