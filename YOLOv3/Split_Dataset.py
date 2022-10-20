import os
import random
import shutil
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


if __name__ == '__main__':

    dataset_dir = os.path.abspath(os.path.join(BASE_DIR, "face_mask_data"))
    # print(dataset_dir)
    split_dir = os.path.abspath(os.path.join(BASE_DIR, "face_mask"))
    train_image_dir = os.path.join(split_dir, "train_images")
    train_label_dir = os.path.join(split_dir, "train_labels_xml")
    # valid_dir = os.path.join(split_dir, "valid")
    test_image_dir = os.path.join(split_dir, "test_images")
    test_label_dir = os.path.join(split_dir, "test_labels_xml")

    if not os.path.exists(dataset_dir):
        raise Exception("\n{} 不存在，请下载 face_mask数据 放到\n{} 下，并解压即可".format(
            dataset_dir, os.path.dirname(dataset_dir)))

    train_pct = 0.8
    # valid_pct = 0.1
    test_pct = 0.2
    state = np.random.get_state()

    for root, dirs, files in os.walk(dataset_dir):
        for sub_dir in dirs:

            imgs = os.listdir(os.path.join(root, sub_dir))#返回的是列表
            if imgs[0].endswith('.jpg'):
                imgs = list(filter(lambda x: x.endswith('.jpg'), imgs))
                imgs = np.array(imgs)
                np.random.set_state(state)
                np.random.shuffle(imgs)
                train_dir = train_image_dir
                test_dir = test_image_dir
            else:
                imgs = list(filter(lambda x: x.endswith('.xml'), imgs))
                imgs = np.array(imgs)
                np.random.set_state(state)
                np.random.shuffle(imgs)
                train_dir = train_label_dir
                test_dir = test_label_dir
            img_count = len(imgs)#列表长度即图片个数

            train_point = int(img_count * train_pct)
            # valid_point = int(img_count * (train_pct + valid_pct))

            for i in range(img_count):
                if i < train_point:
                    out_dir = train_dir
                # elif i < valid_point:
                #     out_dir = os.path.join(valid_dir, sub_dir)
                else:
                    out_dir = test_dir

                makedir(out_dir)

                target_path = os.path.join(out_dir, imgs[i])
                src_path = os.path.join(dataset_dir, sub_dir, imgs[i])

                shutil.copy(src_path, target_path)
            print('Class:{}, train:{}, test:{}'.format(sub_dir, train_point, img_count-train_point))
            print("已在 {} 创建划分好的数据\n".format(out_dir))