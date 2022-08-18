import os
import random
from PIL import Image
from torch.utils.data import Dataset


random.seed(1)
data_label = {"0": "cat", "1": "dog"}

class CatDogDataset(Dataset):
    def __init__(self, data_dir, mode="train", split_n=0.9, rng_seed=111, transform=None):
        """
        猫狗分类的Dataset
        :param data_dir:
        :param mode:
        :param split_n:
        :param rng_seed:
        :param transform:
        """
        self.mode = mode
        self.data_dir = data_dir
        self.rng_seed = rng_seed
        self.split_n = split_n
        self.transform = transform
        self.data_info = self._get_img_info()  # 元素为元组，存储所有图片路径和标签，每个元组是图片的路径和其标签

    def __getitem__(self, idx):
        path_img, label = self.data_info[idx]
        img = Image.open(path_img).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        if len(self.data_info) == 0:
            raise Exception(f"\ndata_dir:{self.data_dir} is empty! Please check out your path of images!")
        return len(self.data_info)

    def _get_img_info(self):
        img_names = os.listdir(self.data_dir)
        img_names = list(filter(lambda x: x.endswith(".jpg"), img_names))

        random.seed(self.rng_seed)
        random.shuffle(img_names)

        # 处理labels
        img_labels = [0 if n.startswith("cat") else 1 for n in img_names]

        # 分割数据集为训练集和验证集
        split_idx = int(len(img_labels) * self.split_n)  # 按照比例分割的idx
        if self.mode == "train":
            img_set = img_names[:split_idx]
            label_set = img_labels[:split_idx]
        elif self.mode == "valid":
            img_set = img_names[split_idx:]
            label_set = img_labels[split_idx:]
        else:
            raise Exception("Only train or valid~~~")
        path_img_set = [os.path.join(self.data_dir, n) for n in img_set]
        data_info = [(n, l) for n, l in zip(path_img_set, label_set)]

        return data_info



















