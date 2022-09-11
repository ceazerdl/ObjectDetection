import torch
import cv2 as cv
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


def nms(boxes: Tensor, scores: Tensor, thr: float, top_k: int):
    """
    :param boxes: Tensor[N, 4] 尚未筛选的boxes
    :param scores:
    :param thr:
    :param top_k:
    :return:
    """
    keep = scores.new(scores.size()).zero_().long()

    if boxes.numel() == 0:
        return keep

    x1 = boxes[:, 0]
    print("x1.shape:{}".format(x1.size()))
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1) * (y2 - y1)
    count = 0
    _, idx = scores.sort()  # 升序排列
    idx = idx[-top_k:]
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    while idx.numel():
        max_score_i = idx[-1]
        keep[count] = max_score_i
        count += 1
        if idx.size() == 1:
            break
        idx = idx[:-1]
        # 取出idx中的剩余框的索引，用来和最大score的框进行比较
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x1, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        xx1 = torch.clamp(xx1, min=x1[max_score_i])
        yy1 = torch.clamp(yy1, min=y1[max_score_i])
        xx2 = torch.clamp(xx2, max=x2[max_score_i])
        yy2 = torch.clamp(yy2, max=y2[max_score_i])
        flag = 1
        if flag:
            print("xx1.shape:{}".format(xx1.size()))
        # 计算wh
        w.resize_as(xx1)
        h.resize_as(yy1)
        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        if flag:
            print("w.shape:{}".format(w.size()))
        inter = w * h
        remain_area = torch.index_select(area, 0, idx)
        union_area = area[max_score_i] + remain_area - inter
        ious = inter / union_area
        if flag:
            print("ious.shape:{}".format(ious.size()))

        idx = idx[ious[0].le(thr)]

    return keep, count


if __name__ == "__main__":
    # boxes_scores = np.array([[100, 100, 210, 210, 0.72],
    #                   [250, 250, 420, 420, 0.8],
    #                   [220, 220, 320, 330, 0.92],
    #                   [100, 100, 210, 210, 0.72],
    #                   [230, 240, 325, 330, 0.81],
    #                   [220, 230, 315, 340, 0.9]])
    boxes_scores = np.array([[100, 100, 210, 210, 0.72],
                             [250, 250, 420, 420, 0.8],
                             [220, 220, 320, 330, 0.92],
                             [100, 400, 240, 240, 0.72],
                             [230, 240, 325, 330, 0.81],
                             [220, 230, 315, 340, 0.9]])
    boxes = boxes_scores[:, :4]
    scores = boxes_scores[:, 4]
    boxes = torch.from_numpy(boxes)
    scores = torch.from_numpy(scores)
    print("boxes.shape:{}".format(boxes.size()))
    keep = torch_nms(boxes, scores, thr=0.3)
    print(keep)
    img = np.zeros((1024, 1024, 3), dtype=np.int8)
    for i in range(len(boxes)):
        img = cv.rectangle(img, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), (np.random.randint(0, 255),
                                                                                         np.random.randint(0, 255),
                                                                                         np.random.randint(0, 255)), 2)
        img = cv.putText(img, str(i), (boxes[i][0], boxes[i][1]), 2, 1, (255, 255, 255), 2)
    for j in range(len(keep)):
        img = cv.rectangle(img, (boxes[keep[j]][0]+400, boxes[keep[j]][1]), (boxes[keep[j]][2]+400, boxes[keep[j]][3]),
                           (np.random.randint(0, 255),
                            np.random.randint(0, 255),
                            np.random.randint(0, 255)), 2)
        img = cv.putText(img, str(keep[j].item()), (boxes[keep[j]][0]+400, boxes[keep[j]][1]), 2, 0.8, (255, 255, 255),
                         2)
    cv.imshow("img", img)
    cv.waitKey(0)
    cv.destroyAllWindows()






















