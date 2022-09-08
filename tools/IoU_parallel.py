import torch
from torch import Tensor
import numpy as np
from numpy import array


# PyTorch实现并行Iou计算
def box_area_pt(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by its (x1, y1, x2, y2) coordinates
    :param boxes: (Tensor[N, 4])
    :return: area: (Tensor[N]) area for each box
    """

    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_area_np(boxes: array) -> array:
    """
    :param boxes:
    :return: area
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou_pt(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are excepted to be in (x1, y1, x2, y2) format.
    :param boxes1: Tensor[N, 4]
    :param boxes2: Tensor[N, 4]
    :return: iou: Tensor[N, M] the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area_pt(boxes1)
    area2 = box_area_pt(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]  N中每一个和M中每一个比较
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # NxMx2
    # wh = rb - lt
    # wh = torch.clamp(wh, min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]  # NxM

    iou = inter / (area1[:, None] + area2 - inter)  # broadcast  NxM

    return iou


def box_iou_np(boxes1: array, boxes2: array) -> array:

    area1 = box_area_np(boxes1)
    area2 = box_area_np(boxes2)

    lt = np.maximum(boxes1[:, np.newaxis, :2], boxes2[:, :2])
    rb = np.maximum(boxes1[:, np.newaxis, 2:], boxes2[:, 2:])

    wh = (rb - lt).clip(min=0)
    # wh = rb - lt
    # wh = np.clip(wh, min=0)
    # wh = np.maximum(0, wh)
    inter = wh[:, :, 0] * wh[:, :, 1]

    iou = inter / (area1[:, np.newaxis] + area2 - inter)

    return iou


if __name__ == "__main__":
    pass
















