import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader


random.seed(111)
class_name = ["ALB", "BET", "DOL", "LAG", "NoF", "OTHER", "SHARK", "YFT"]


class NCFMDateSet(Dataset):

    def __init__(self, data_dir, mode="train", split_n=0.9, rng_seed=111, transform=None):
        super(NCFMDateSet, self).__init__()
        self.mode = mode
        self.data_dir = data_dir
        self.rng_seed = rng_seed
        self.split_n = split_n
        self.transform = transform
        self.data_info = self.get_img_info()

    def __getitem__(self, idx):
        img_path, label = self.data_info[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        if len(self.data_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(self.data_dir))
        return len(self.data_info)

    def get_img_info(self):
        img_path = list()
        for root, dirs, files in os.walk(self.data_dir):
            for name in files:
                if name.endswith(".jpg"):
                    img_path.append(os.path.join(root, name))

        random.seed(self.rng_seed)
        random.shuffle(img_path)
        # list.index(x[, start[, end]]) 函数用于从列表中找出某个值第一个匹配项的索引位置，strat和end是起始和结束位置
        # dirname功能：去掉文件名，返回目录
        # basename功能：返回path最后的文件名
        img_labels = [class_name.index(os.path.basename(os.path.dirname(p))) for p in img_path]

        split_idx = int(len(img_labels) * self.split_n)
        if self.mode == "train":
            imgs_set = img_path[:split_idx]
            labels_set = img_labels[:split_idx]
        elif self.mode == "valid":
            imgs_set = img_path[split_idx:]
            labels_set = img_labels[split_idx:]
        else:
            raise Exception("Mode only supports train and valid~~~")
        data_info = [(n, l) for n, l in zip(imgs_set, labels_set)]
        return data_info


if __name__ == "__main__":
    dataset = NCFMDateSet("NCFM-train", mode="valid")
    print(dataset.data_info)
