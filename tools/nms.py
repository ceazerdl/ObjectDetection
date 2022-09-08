import torch
from torch import Tensor
import numpy as np
from numpy import array
from IoU_parallel import box_iou_pt, box_iou_np


def torch_nms(boxes: Tensor, scores: Tensor, thr: float) -> Tensor:
    """
    :param boxes: [N, 4] 此处传进来的框，是经过筛选(NMS之前选取过得分TopK)之后，在传入之前处理好的
    :param scores: [N]
    :param thr: 0.7
    :return: Tensor
    """
    keep = []
    idxs = scores.argsort()  # 升序排列

    while idxs.numel() > 0:
        # 取出最高分数
        max_score_idx = idxs[-1]
        max_score_box = boxes[max_score_idx][None, :]
        keep.append(max_score_idx)
        if idxs.size() == 1:
            break
        idxs = idxs[:-1]
        other_box = boxes[idxs]
        ious = box_iou_pt(max_score_box, other_box)  # 1个框和其余框比较 1xM
        idxs = idxs[ious[0] <= thr]

    keep = idxs.new(keep)
    return keep


def np_nms(boxes: array, scores: array, thr: float) -> array:
    idxs = scores.argsort()  # 升序排列

    keep = []
    while idxs.size():
        max_score_idx = idxs[-1]
        max_score_box = boxes[max_score_idx][np.newaxis, :]
        keep.append(max_score_idx)
        if idxs.size() == 1:
            break
        idxs = idxs[:-1]
        other_box = boxes[idxs]
        ious = box_iou_np(max_score_box, other_box)
        idxs = idxs[ious[0] <= thr]

    keep = np.array(keep)
    return keep


if __name__ == "__main__":
    pass





















