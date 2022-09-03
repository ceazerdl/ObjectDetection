import cv2
import numpy as np


def Compute_IoU(box_1, box_2):
    '''计算两个矩形框的iou'''
    #计算两个矩形框面积之和，不是并集
    area_1 = (box_1[2] - box_1[0]) * (box_1[3] - box_1[1])
    area_2 = (box_2[2] - box_2[0]) * (box_2[3] - box_2[1])
    sum_area = area_1 + area_2
    #计算两个框的交集，左上角的点取两个框中大的值，右下角取小的那个值
    LeftTop_x = max(box_1[0], box_2[0])
    LeftTop_y = max(box_1[1], box_2[1])
    RightBottom_x = min(box_1[2], box_2[2])
    RightBottom_y = min(box_1[3], box_2[3])

    #判断一下，是不是交集
    if LeftTop_x >= RightBottom_x or LeftTop_y >= RightBottom_y:
        return 0
    else:
        in_sec = (RightBottom_x - LeftTop_x) * (RightBottom_y - LeftTop_y)
        print("交集面积是：%s" %in_sec)
        iou_value = in_sec / (sum_area - in_sec)
        print(f"iou的值是{iou_value}")
        return iou_value

def draw_box(box1, box2):
    '''绘制框框'''
    #模拟一张图片,生成一张空的三通道图片
    img = np.zeros((300, 300, 3), np.uint8)
    print(img.shape)

    #使用左上和右下两个坐标在图片上绘制box1
    LT_box_1 = tuple(box1[:2])
    RB_box_1 = tuple(box1[2:])
    point_color_1 = (0, 255, 0) #BGR
    thickness_1 = 3
    lineType_1 = 4
    cv2.rectangle(img, LT_box_1, RB_box_1, point_color_1, thickness_1, lineType_1)

    #绘制box2
    LT_box_2 = tuple(box2[:2])
    RB_box_2 = tuple(box2[2:])
    point_color_2 = (0, 0, 255)  # BGR
    thickness_2 = 3
    lineType_2 = 4
    cv2.rectangle(img, LT_box_2, RB_box_2, point_color_2, thickness_2, lineType_2)

    # cv2.namedWindow("IOU")
    cv2.imshow("IOU", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    rec1 = [0, 0, 200, 200]
    rec2 = [0, 0, 200, 100]
    draw_box(rec1, rec2)
    iou = Compute_IoU(rec1, rec2)
