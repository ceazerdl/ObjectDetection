import glob
import os
import numpy as np
from skimage import io


# left top right bottom 2 cx cy width height
def corner2center(corner, w, h):
    if len(corner.shape) == 1:
        corner = np.expand_dims(corner, 0)
    center_xy = (corner[:, :2] + corner[:, 2:])/2
    center_wh = corner[:, 2:] - corner[:, :2]
    center = np.hstack([center_xy, center_wh])
    center[:, ::2] /= w
    center[:, 1::2] /= h
    return center


'''
# cx cy width height 2 left top right bottom
def center2corner(center, w, h):
    if len(center.shape) == 1:
        center = np.expand_dims(center, 0)
    center[:, ::2] *= w
    center[:, 1::2] *= h
    corner_lt = center[:, :2] - center[:, 2:] / 2
    corner_rb = center[:, :2] + center[:, 2:] / 2
    corner = np.hstack([corner_lt, corner_rb])
    return corner
    '''


if __name__ == '__main__':

    in_path = r'face_mask\\train_labels_wh'
    img_list = glob.glob(in_path + '/*.txt')
    ignore_txt = []
    for img_file in img_list:
        data = np.loadtxt(img_file, ndmin=2)
        # 此处data包含w, h, cata_id, x1, y1, x2, y2
        if len(data) == 0:
            # ignore_txt.append(img_file)
            continue
        else:
            # if data[0, 2] <= 1:
            #     data = np.hstack([data[:, 0, None], center2corner(data[:, 1:], w, h)])
            # else:
            label_data = np.hstack([data[:, 2, None], corner2center(data[:, 3:], data[0, 0], data[0, 1])])
            np.savetxt(img_file.replace("train_labels_wh", "train_labels"), label_data, fmt='%d %f %f %f %f')
    # print(ignore_txt)