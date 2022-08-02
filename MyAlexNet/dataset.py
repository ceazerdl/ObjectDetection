from torch.utils.data import Dataset
import os
import random
import cv2
from PIL import Image


random.seed(1)
data_label = {"dog": 1, "cat": 0}


class CatAndDogDataset(Dataset):
    def __init__(self, data_dir, mode="train", split_n=0.9, rng_seed=111, transform=None):
        '''
        猫狗数据集分类任务的DataSet
        :param data_dir: 文件路径
        :param mode: 当前模式
        :param split_n: 训练集和验证集的划分比例
        :param rng_seed: 随机种子
        :param transform: 数据预处理
        '''
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        self.split_n = split_n
        self.rng_seed = rng_seed
        # 这是个列表，该列表里的元素是元组，每个元组里保存一张图片的路径和该图片的标签
        self.data_info = self.get_img_info()

    def __getitem__(self, index):
        '''
        这个方法尽可能的简单，能不在这里面做的，就不要在这里面做，否则加载数据的过程会非常慢
        :param index:
        :return:
        '''
        path_img, label = self.data_info[index]
        # img = cv2.imread(path_img)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.open(path_img).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        '''
        返回数据集长度
        :return:
        '''
        if len(self.data_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path of images!".format(self.data_dir))
        return len(self.data_info)

    def get_img_info(self):
        img_names = os.listdir(self.data_dir)
        img_names = list(filter(lambda x: x.endswith(".jpg"), img_names))

        # 这个方法会使用两次，一次是在生成训练集信息时，一次是在生成验证集信息时，所以为了保证两次数据集划分的一致性，需要固定random_seed
        random.seed(self.rng_seed)
        random.shuffle(img_names)

        # 处理labels
        img_labels = [0 if name.startswith("cat") else 1 for name in img_names]

        # 划分训练集和验证集
        split_idx = int(len(img_labels) * self.split_n)
        if self.mode == "train":
            img_set = img_names[:split_idx]
            label_set = img_labels[:split_idx]
        elif self.mode == "valid":
            img_set = img_names[split_idx:]
            label_set = img_labels[split_idx:]
        else:
            raise Exception("self.mode 无法识别，仅支持train和valid两种")

        path_img_set = [os.path.join(self.data_dir, name) for name in img_set]
        data_info = [(name, label) for name, label in zip(path_img_set, label_set)]

        return data_info






















