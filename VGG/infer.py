import os
os.environ["NLS_LANG"] = "SIMPLIFIED CHINESE_CHINA.UTF8"
import time
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from common_tools import get_vgg16


base_dir = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name())
print(torch.cuda.device_count())

def img_transform(img_rgb, transform=None):
    """
    将数据转换为模型的输入tensor
    :param img_rgb:
    :param transform:
    :return:
    """
    if transform is None:
        raise Exception("No transform is not permitted~~~")
    img = transform(img_rgb)
    return img

def process_img(path_img):
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    infernece_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    img_rgb = Image.open(path_img).convert("RGB")
    img_tensor = img_transform(img_rgb, infernece_transform)
    img_tensor.unsqueeze_(0)
    img_tensor = img_tensor.to(device)
    return img_tensor, img_rgb

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


if __name__ == "__main__":
    path_state_dict = os.path.join(base_dir, "data", "vgg16-397923af.pth")
    path_img = os.path.join(base_dir, "data", "Golden Retriever from baidu.jpg")
    path_classnames = os.path.join(base_dir, "data", "imagenet1000.json")
    path_classnames_cn = os.path.join(base_dir, "data", "imagenet_classnames.txt")

    cls_n, cls_n_cn = load_class_names(path_classnames, path_classnames_cn)

    img_tensor, img_rgb = process_img(path_img)

    vgg_model = get_vgg16(path_state_dict, device, True)

    with torch.no_grad():
        start_time = time.time()
        outputs = vgg_model(img_tensor)
        infer_time = time.time() - start_time

    _, pred_int = torch.max(outputs.data, 1)
    print(pred_int)
    _, top5_idx = torch.topk(outputs.data, 5, dim=1)
    print(top5_idx)

    pred_idx = int(pred_int.cpu().numpy())
    pred_str, pred_cn = cls_n[pred_idx], cls_n_cn[pred_idx]
    print(f"img:{os.path.basename(path_img)} is: {pred_str}\n {pred_cn}")
    print("time consumign:{:.2f}s".format(infer_time))

    plt.imshow(img_rgb)
    plt.title("predict:{}".format(pred_str))
    top5_num = top5_idx.cpu().numpy().squeeze()
    text_str = [cls_n[t] for t in top5_num]
    for idx in range(len(top5_num)):
        plt.text(5, 15+idx*30, "top {}: {}".format(idx+1, text_str[idx]), bbox=dict(fc="magenta"))

    plt.show()



































