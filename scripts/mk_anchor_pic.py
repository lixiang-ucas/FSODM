import cv2
import os
import sys
import math
import numpy as np


def drawline(img, pt1, pt2, color, thickness=1, style='dotted', gap=20):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    if style == 'dotted':
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    else:
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1


if __name__ == '__main__':
    # img_path = sys.argv[1]
    # anno_path = sys.argv[2]
    img_path = '../../20726.jpg'
    anno_path = '../../20726__1__0___0.txt'

    with open(anno_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split()
            if line[-1] == 'airplane':
                bbox = [float(line[0]), float(line[1]), float(line[2]), float(line[3])]
                break

    center_x = (bbox[2] + bbox[0]) / 2
    center_y = (bbox[3] + bbox[1]) / 2

    img = cv2.imread(img_path)

    shape = img.shape
    for ind, inter in enumerate([32, 16, 8]):
        img_out = img.copy()
        for x in range(0, shape[1], inter):
            cv2.line(img_out, (x, 0), (x, shape[0]), (0, 0, 0), 1, 1)
        cv2.line(img_out, (shape[1] - 1, 0), (shape[1] - 1, shape[0]), (0, 0, 0), 1, 1)
        for x in range(0, shape[0], inter):
            cv2.line(img_out, (0, x), (shape[1], x), (0, 0, 0), 1, 1)
        cv2.line(img_out, (0, shape[0] - 1), (shape[1], shape[0] - 1), (0, 0, 0), 1, 1)

        cv2.rectangle(img_out, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)

        center_left = inter * (center_x // inter)
        center_top = inter * (center_y // inter)

        if inter == 8:
            anchors = [(10, 13), (16, 30), (33, 23)]
            thick = 1
        elif inter == 16:
            thick = 2
            anchors = [(30, 61), (62, 45), (59, 119)]
        else:
            thick = 2
            anchors = [(116, 90), (156, 198), (373, 326)]

        anchor_center = (center_left + inter / 2, center_top + inter / 2)
        for i, anchor in enumerate(anchors):
            cv2.rectangle(img_out,
                         (int(anchor_center[0] - anchor[0] / 2), int(anchor_center[1] - anchor[1] / 2)),
                         (int(anchor_center[0] + anchor[0] / 2), int(anchor_center[1] + anchor[1] / 2)),
                         (0, 255, 255),
                         thick)
        # cv2.rectangle(img_out, (center_left, center_top), (center_left + inter, center_top + inter), (0, 0, 255), thick)
        if inter == 8:
            sub_img_left = int(center_left + inter / 2 - 50)
            sub_img_top = int(center_top + inter / 2 - 50)
            sub_img = img_out[sub_img_top:sub_img_top + 100, sub_img_left:sub_img_left + 100]
            sub_img = cv2.resize(sub_img, (400, 400))
            img_out[shape[1] - 450:shape[1] - 50, shape[0] - 450:shape[0] - 50] = sub_img
            cv2.rectangle(img_out,
                         (sub_img_left - 1, sub_img_top - 1),
                         (sub_img_left + 101, sub_img_top + 101),
                         (0, 0, 255),
                         3)
            cv2.rectangle(img_out,
                         (shape[1] - 451, shape[0] - 451),
                         (shape[1] - 49, shape[0] - 49),
                         (0, 0, 255),
                         3)
            drawline(img_out,
                    (sub_img_left, sub_img_top + 101),
                    (shape[1] - 451, shape[0] - 50),
                    (0, 0, 255),
                    4,
                    gap=15)

            drawline(img_out,
                     (sub_img_left + 101, sub_img_top),
                     (shape[1] - 50, shape[0] - 451),
                     (0, 0, 255),
                     4,
                     gap=15)

        cv2.imwrite(f'../../anchor{ind + 1}.jpg', img_out)


