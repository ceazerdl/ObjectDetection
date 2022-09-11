import torch
import cv2 as cv
from torch import Tensor
from IoU_parallel import *


def soft_nms(boxes_scores: Tensor, score_thr: float, iou_thr: float, weight_method=2, sigma=0.3) -> Tensor:
    """
    与nms的区别：
    1. nms直接根据iou_thr筛选boxes，softnms更新分数，然后根据score_thr更新boxes_scores
    2. nms只需要在循环外的一次排序，使用排序号的索引进行操作，softnms需要每次在循环内找到最大值的索引，
    然后手动交换最大值和最后一个数据的位置，然后将最后一个数据去掉，以减少boxes_scores，从而正常求取ious
    3. 由于softnms会变更筛选出来的最大值和最后一个数据的位置，所以将最大值放入list中时需要深拷贝，不然放入到list中的值会因为交换而改变
    :param boxes_scores:
    :param score_thr:
    :param iou_thr:
    :param weight_method:
    :param sigma:
    :return:
    """
    keep = list()   # 存储bboxes和scores
    while boxes_scores.size(0):
        max_score_i = torch.argmax(boxes_scores[:, -1])
        max_score_box = boxes_scores[max_score_i].clone()       # 这个地方非常重要
        keep.append(max_score_box)
        if boxes_scores.size(0) == 1:      # 这个地方要写明确哪个维度为1
            break
        max_box = max_score_box[:-1]
        boxes_scores[max_score_i] = boxes_scores[-1]    # 和最后一行交换
        boxes_scores = boxes_scores[:-1]                # 舍弃掉最后一行
        ious = box_iou_pt(max_box.unsqueeze(0), boxes_scores[:, :-1])

        if weight_method == 1:
            # 关键问题在于更新分数时，保持赋值两边索引的一致性
            # ious是和boxes_scores一致的，将其中大于iou_thr的索引出来，然后更新权重，在boxes_scores中对应位置更新scores
            boxes_scores[:, -1][ious[0].gt(iou_thr)] *= 1 - ious[0][ious[0].gt(iou_thr)]
            boxes_scores = boxes_scores[boxes_scores[:, -1].ge(score_thr)]
        elif weight_method == 2:
            boxes_scores[:, -1] = boxes_scores[:, -1] * torch.exp(-ious * ious / sigma)
            boxes_scores = boxes_scores[boxes_scores[:, -1].ge(score_thr)]
        else:
            # hard nms
            # 不能用squeeze函数，因为ious的维度为1时会被压缩掉
            # print(f"inter---4---:{boxes_scores.size()}")
            # print(f"inter---5---:{ious.squeeze().le(iou_thr)}")
            # print(f"inter---6---:{ious.squeeze().le(iou_thr).size()}")
            # print(f"inter---7---:{ious[0].le(iou_thr)}")
            # print(f"inter---8---:{ious[0].le(iou_thr).size()}")
            boxes_scores = boxes_scores[ious[0].le(iou_thr)]

    keep = torch.stack(keep) if len(keep) else torch.tensor([])
    return keep


if __name__ == "__main__":
    boxes_scores_1 = np.array([[100, 100, 210, 210, 0.1],
                      [250, 250, 420, 420, 0.8],
                      [220, 220, 320, 330, 0.92],
                      [100, 100, 240, 240, 0.72],
                      [230, 240, 325, 350, 0.81],
                      [250, 250, 315, 340, 0.9]])

    boxes_scores = np.array([[100, 100, 210, 210, 0.72],
                             [250, 250, 420, 420, 0.8],
                             [220, 220, 320, 330, 0.92],
                             [100, 400, 240, 240, 0.72],
                             [230, 240, 325, 330, 0.81],
                             [220, 230, 315, 340, 0.9]])
    boxes_scores = torch.from_numpy(boxes_scores)
    boxes_scores_2 = boxes_scores.clone()
    # 方法1和方法2主要调整score_thr，hard nms只调整iou_thr.可以先调整iou_thr，再调整score_thr
    keep = soft_nms(boxes_scores, score_thr=0.3, iou_thr=0.3, weight_method=2, sigma=0.3)
    img = np.zeros((1024, 1024, 3), dtype=np.int8)
    for i in range(len(boxes_scores)):

        img = cv.rectangle(img, (boxes_scores_2[i][0], boxes_scores_2[i][1]), (boxes_scores_2[i][2], boxes_scores_2[i][3]),
                           (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)), 2)
        img = cv.putText(img, str(i), (boxes_scores_2[i][0]+i*10, boxes_scores_2[i][1]), 2, 1, (255, 0, 255), 1)
    for j in range(len(keep)):

        img = cv.rectangle(img, (keep[j][0] + 400, keep[j][1]),
                           (keep[j][2] + 400, keep[j][3]),
                           (np.random.randint(0, 255),
                            np.random.randint(0, 255),
                            np.random.randint(0, 255)), 2)
        img = cv.putText(img, str(j), (keep[j][0] + 400, keep[j][1]), 2, 0.8,
                         (255, 255, 255),
                         2)
    cv.imshow("img", img)
    cv.waitKey(0)
    cv.destroyAllWindows()






















