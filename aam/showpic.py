import os
import re

import numpy as np
import cv2
from parse import parse

ROOT_PATH = "/Users/xc5/PycharmProjects/hipjoint/aam/1"

def read_pts_vector(pts_f):
    pts_f.readline()
    pts_f.readline()
    pts_f.readline()
    oneline = pts_f.readline()
    line_count = 0
    all_vec = []
    while oneline:
        line_count += 1
        if (re.match("[ \t]*\}", oneline)):
            break
        strarrs = oneline.split(" ")
        x = int(strarrs[0])
        y = int(strarrs[1])
        all_vec.append([x, y])
        oneline = pts_f.readline()
        pass
    return all_vec

def display_pic(pic_file:str, pts_file:str):
    with open(pts_file) as pts_f:
        pts_vec = read_pts_vector(pts_f)

        # Create a black image
        img = np.zeros((512, 512, 3), np.uint8)

        # Draw the image
        img = cv2.imread(pic_file,0)

        # Draw a diagonal blue line with thickness of 5 px
        for i in range(0, int(len(pts_vec))):
            img = cv2.circle(img, (pts_vec[i][0], pts_vec[i][1]), 5, (0, 0, 255), -1)
            if (i <= len(pts_vec) - 2):
                img = cv2.line(img,
                               (pts_vec[i][0], pts_vec[i][1]),
                               (pts_vec[i+1][0], pts_vec[i+1][1]),
                               (255 - i * 2, 0, 0), 5)

        for i in range(10):
            print("Metric ", i,  " = ", calc(i, pts_vec))

        img = cv2.putText(img, 'Metric 2 = %s' % str(calc(2, pts_vec)), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 10, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('image', img)
        cv2.waitKey(0)

def calc(metric:int, pts:list):
    if metric == 1:
        return pow(pow(pts[4][0] - pts[26][0], 2) + pow(pts[4][1] - pts[26][1], 2), 1.0 / 2)
    if metric == 2:
        return pow(pow(pts[4][0] - pts[26][0], 2) +  pow(pts[4][1] - pts[26][1], 2), 1.0/2)
    else:
        return 0

def batch_process_jpg(root_path):
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".jpg"):
                display_pic(os.path.join(root,file), os.path.join(root,file.replace(".jpg", ".pts")))
                pass


if __name__ == "__main__":
    flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
    print(flags)
    batch_process_jpg(ROOT_PATH)