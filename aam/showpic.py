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

def point_distance(pts, n1:int, n2:int):
    return pow(pow(pts[n1][0] - pts[n2][0], 2) + pow(pts[n1][1] - pts[n2][1], 2), 0.5)

def point_to_line_distance(pts, x3:int, y3:int, line_n1, line_n2):
    '''
    输入两点，建立直线方程y＝kx＋b。
    输入第3点，求点到直线的距离。
    '''

    # 输入两点p1, p2坐标
    # sys.stdout.write('Input two points:\n')
    # line = sys.stdin.readline()
    [x1, y1] = pts[line_n1]
    [x2, y2] = pts[line_n2]

    if (x2 == x1):
        raise Exception("Cannot divide zero")

    # 计算k,b
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1

    # 计算点p3到直线距离
    # sys.stdout.write('The dictionary is:\n')
    d = abs(k * x3 - y3 + b) / ((-1) * (-1) + k * k) ** 0.5
    return d

def point_n_to_line_distance(pts, point_n, line_n1, line_n2):
    import sys

    '''
    输入两点，建立直线方程y＝kx＋b。
    输入第3点，求点到直线的距离。
    '''

    # 输入两点p1, p2坐标
    # sys.stdout.write('Input two points:\n')
    # line = sys.stdin.readline()
    [x1, y1] = pts[line_n1]
    [x2, y2] = pts[line_n2]

    if (x2 == x1):
        raise Exception("Cannot divide zero")

    # 计算k,b
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1

    # 输入第三点p3坐标
    # sys.stdout.write('Input the third point:\n')
    # line = sys.stdin.readline()
    [x3, y3] = pts[point_n]

    # 计算点p3到直线距离
    # sys.stdout.write('The dictionary is:\n')
    d = abs(k * x3 - y3 + b) / ((-1) * (-1) + k * k) ** 0.5
    return d


def get_middle_between_points(pts, n1, n2):
    xm = 0.5 * (pts[n1][0] + pts[n2][0])
    ym = 0.5 * (pts[n1][1] + pts[n2][1])
    return [xm, ym]

def calc(metric:int, pts:list):
    if metric == 1:
        [xa, ya] = get_middle_between_points(pts, 3, 25)
        distance1 = point_to_line_distance(pts, xa, ya, 11, 16)
        return distance1
    if metric == 2:
        return point_distance(pts, 4, 26)
    if metric == 3:
        [xa, ya] = get_middle_between_points(pts, 3, 25)
        dist1 = point_to_line_distance(pts, xa, ya, 11, 16)
        dist2 = point_distance(pts, 4, 26)
        return dist1 / dist2
    if metric == 4:
        [xa, ya] = get_middle_between_points(pts, 2, 24)
        distance3 = point_to_line_distance(pts, xa, ya, 11, 16)
        [xb, yb] = get_middle_between_points(pts, 1, 23)
        distance4 = point_to_line_distance(pts, xb, yb, 11, 16)
        return [distance3, distance4]
    if metric == 5:
        [xa, ya] = get_middle_between_points(pts, 4, 26)
        return point_to_line_distance(pts, xa, ya, 11, 16)
    if metric == 6:
        [xa, ya] = get_middle_between_points(pts, 6, 21)
        return point_to_line_distance(pts, xa, ya, 11, 16)
    if metric == 7:
        return [point_n_to_line_distance(pts, 5, 6, 21),
                point_n_to_line_distance(pts, 27, 6, 21)]
    if metric == 8:
        [xa, ya] = get_middle_between_points(pts, 12, 15)
        distance9 = point_to_line_distance(pts, xa, ya, 11, 16)
        [xb, yb] = get_middle_between_points(pts, 13, 14)
        distance10 = point_to_line_distance(pts, xb, yb, 11, 16)
        return [distance9, distance10]
    if metric == 9:
        [xa, ya] = get_middle_between_points(pts, 7, 20)
        distance11 = point_to_line_distance(pts, xa, ya, 11, 16)
        [xb, yb] = get_middle_between_points(pts, 9, 19)
        distance12 = point_to_line_distance(pts, xb, yb, 11, 16)
        return [distance11, distance12]
    if metric == 10:
        return point_distance(pts, 0, 22)
    if metric == 11:
        [xa, ya] = get_middle_between_points(pts, 1, 23)
        return point_to_line_distance(pts, xa, ya, 8, 19)
    if metric == 12:
        dist13 = point_distance(pts, 0, 22)
        [xa, ya] = get_middle_between_points(pts, 1, 23)
        dist14 = point_to_line_distance(pts, xa, ya, 8, 19)
        return dist14 / dist13
    if metric == 13:
        # 17, 18
        width = 0
        return width
    if metric == 14:
        [xa, ya] = get_middle_between_points(pts, 9, 10)
        left = point_to_line_distance(pts, xa, ya , 7, 20)
        [xa, ya] = get_middle_between_points(pts, 17, 18)
        right = point_to_line_distance(pts, xa, ya , 7, 20)
        return [left, right]
    else:
        return 0

    """
    point num[0] to num[27]
    1.骨盆总高度 distance1 = mid of(num[3],num[25]) to line(num[11],num[16])
    2.骨盆宽度   distance2 = num[4] to num[26]
    3.骨盆高宽比 distance1/distance2
    4.骶髂关节：上高度和下高度（骨盆下顶点连线作为基准水平高度）
    distance3 = mid of (num[2],num[24]) to line(num[11],num[16])
    distance4 = mid of (num[1],num[23]) to line(num[11],num[16])
    5.髂前上棘高度 distance5 = mid of (num[4],num[26]) to line(num[11],num[16])
    6.泪滴高度 distance6 = mid of (num[6],num[21]) to line(num[11],num[16])
    7.左侧和右侧髋臼高度
    distance7 = num[5]  to line(num[6],num[21])
    distance8 = num[27] to line(num[6],num[21])
    8.耻骨联合上高度，下高度
    distance9 = mid of (num[12],num[15]) to line(num[11],num[16])
    distance10 = mid of (num[13],num[14]) to line(num[11],num[16])
    9.闭孔上高度，闭孔下高度
    distance11 = mid of (num[7],num[20]) to line(num[11],num[16])
    distance12 = mid of (num[9],num[19]) to line(num[11],num[16])
    10.骨盆口的宽度
    distance13 = num[0] to num[22]
    11.骨盆口的高度
    distance14 = mid of (num[1] to num[23]) to line(num[8],num[19])
    12.骨盆口的高宽比
    distance14/distance13
    13.左右侧闭孔的宽度
    distance15 = mid of (num[7],num[10]) to line(num[8],num[9])
    distance16 = mid of (num[17],num[20]) to line(num[18],num[19])
    14.左右侧闭孔的高宽比例
    左右侧闭孔高度
    distance17 = mid of (num[7],num[8]) to line(num[9],num[10])
    distance18 = mid of (num[19],num[20) to line(num[17],num[18])
    左右侧高宽比
    distance17/distance15
    distance18/diatance16
    """

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