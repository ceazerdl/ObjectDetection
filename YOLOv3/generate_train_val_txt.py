# -*- coding: utf-8 -*-

import glob
import numpy as np

img_folder = "data/custom/images/"


img_files = glob.glob(img_folder + "*.jpg")
img_num = len(img_files)
shuffle_index = np.random.permutation(img_num)

with open('valid.txt', 'w') as train_txt:
    for each in shuffle_index[:300]:
        train_txt.write(img_files[each].replace('\\', '/') + '\n')

with open('train.txt', 'w') as valid_txt:
    for each in shuffle_index[300:]:
        valid_txt.write(img_files[each].replace('\\', '/') + '\n')


