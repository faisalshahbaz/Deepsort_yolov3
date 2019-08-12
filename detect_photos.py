#! /usr/bin/env python
# -*- coding: utf-8 -*-
# 这个程序用来处理照片，提取特征

from __future__ import division, print_function, absolute_import

import numpy as np
from yolov3_tf import YOLOV3
from Tk import Gui
from judge import JUDGE

from timeit import time
import warnings
import cv2

# from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definition of the parameters
path = "data/photos/"
max_cosine_distance = 0.3
nn_budget = None
nms_max_overlap = 0.3
# f_data = open("train_data.txt", "w")
# f_label = open("train_label.txt", "w")

# deep_sort
model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)

metric = nn_matching.NearestNeighborDistanceMetric("cosine",
                                                   max_cosine_distance,
                                                   nn_budget)
tracker = Tracker(metric)

train_data = []
train_label = []

# 检测处理
for filename in os.listdir(path):
    frame = cv2.imread(path + filename)
    # box 需要 x, y, w, h的格式
    x = 0
    y = 0
    w = frame.shape[0]
    h = frame.shape[1]
    box = [x, y, w, h]

    feature = encoder(frame, [box])
    train_data.append(feature[0, :])
    # print(feature)
    print(filename)
    if filename.find('fall') != -1:
        print('fall!')
        train_label.append(0)
    elif filename.find('stand') != -1:
        print('stand!')
        train_label.append(1)

# PCA
pca = PCA(n_components=3)  # 2dimensions
train_min_data = pca.fit_transform(train_data)
print(train_min_data)

# x = cv2.UMat(train_min_data)
# y = cv2.UMat(train_label)

# 可视化
red_x, red_y, red_z = [], [], []
blue_x, blue_y, blue_z = [], [], []
for i in range(len(train_min_data)):
    if train_label[i] == 1:
        red_x.append(train_min_data[i][0])
        red_y.append(train_min_data[i][1])
        red_z.append(train_min_data[i][2])
    else:
        blue_x.append(train_min_data[i][0])
        blue_y.append(train_min_data[i][1])
        blue_z.append(train_min_data[i][2])
# plt.scatter(red_x, red_y, c='r', marker='x')
# plt.scatter(blue_x, blue_y, c='b', marker='D')
# plt.show()

fig = plt.figure()
ax = Axes3D(fig)
ax.set_title("3D")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.scatter(red_x, red_y, red_z, c='r')
ax.scatter(blue_x, blue_y, blue_z, c='b')
plt.show()

# SVM
svm = cv2.ml.SVM_create()  # create
# 属性设置
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setC(0.01)
# 训练

result = svm.train(train_data, cv2.ml.ROW_SAMPLE, train_label)

# f_data.write(str(train_data))
# f_label.write(str(train_label))

# f_data.close()
# f_label.close()