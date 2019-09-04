#! /usr/bin/env python
# -*- coding: utf-8 -*-
# 这个程序用来处理照片，提取特征

from __future__ import division, print_function, absolute_import

import cv2
from numpy import random
import numpy as np
# from yolo import YOLO

# 导入TensorFlow和tf.keras
import tensorflow as tf
from tensorflow import keras

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import os
from sklearn.model_selection import train_test_split

# Definition of the parameters
path_fall = "/home/tom/桌面/行人检测算法/people/Fall_Augmentation/"
path_upright = "/home/tom/桌面/行人检测算法/people/Upright_Augmentation/"
max_cosine_distance = 0.3
nn_budget = None
nms_max_overlap = 0.3
fall_data = []
upright_data = []

X = []
Y = []

#======================= 特征提取处理样本的代码 ============================
# model_filename = 'model_data/mars-small128.pb'
# encoder = gdet.create_box_encoder(model_filename, batch_size=1)

# metric = nn_matching.NearestNeighborDistanceMetric("cosine",
#                                                    max_cosine_distance,
#                                                    nn_budget)
# tracker = Tracker(metric)

# num = 0
# for filename in os.listdir(path_fall):
#     frame = cv2.imread(path_fall + filename)
#     # box 需要 x, y, w, h的格式
#     x = 0
#     y = 0
#     w = frame.shape[1]
#     h = frame.shape[0]
#     box = [x, y, w, h]

#     feature = encoder(frame, [box])
#     # fall_data.append(feature[0, :])
#     X.append(feature[0, :])
#     Y.append(1)  # 跌倒是1，第二个数字大跌倒
#     num += 1
#     print(num)

# num = 0
# for filename in os.listdir(path_upright):
#     frame = cv2.imread(path_upright + filename)
#     # box 需要 x, y, w, h的格式
#     x = 0
#     y = 0
#     w = frame.shape[1]  # 这里的顺序一定要注意，最开始我就写错了
#     h = frame.shape[0]
#     box = [x, y, w, h]

#     feature = encoder(frame, [box])
#     # upright_data.append(feature[0, :])
#     X.append(feature[0, :])
#     Y.append(0)
#     num += 1
#     print(num)

# np.savetxt("X.txt", X)
# np.savetxt("Y.txt", Y)

# =======================读取数据=====================

X = np.loadtxt("X.txt")
Y = np.loadtxt("Y.txt")

X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.2,
                                                    random_state=42)
# random_state 保证每次随机完结果都一样
# ===================================================

# ======================训练模型=======================
model = keras.Sequential()
model.add(keras.layers.Dense(units=20, activation='relu', input_dim=128))
model.add(keras.layers.Dense(units=2, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=200)

test_loss, test_acc = model.evaluate(X_test, Y_test)
print('Test accuracy:', test_acc)
model.save('fall_detec_model.h5')

# ================================================================================

# ========================测试代码=========================
# model = keras.models.load_model('fall_detec_model_09475.h5')
# input_shape = (1, 128)
# model.build(input_shape)
# model.summary()
# pre_y = model.predict(X_test)
# print(Y_test)
# print(pre_y)
# =======================================================
