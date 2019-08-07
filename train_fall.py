#! /usr/bin/env python
# -*- coding: utf-8 -*-
# 这个程序用来处理照片，提取特征

from __future__ import division, print_function, absolute_import

import cv2
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
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definition of the parameters
path = "data/photos/"
max_cosine_distance = 0.3
nn_budget = None
nms_max_overlap = 0.3

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

train_array = np.array(train_data[0:34])
train_labels = np.array(train_label[0:34])

test_array = np.array(train_data[34:])
test_labels = np.array(train_label[34:])

# model = keras.Sequential([
#     keras.layers.Dense(10, activation=tf.nn.relu),
#     keras.layers.Dense(2, activation=tf.nn.softmax)
# ])

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(train_array, train_labels, epochs=250)

# test_loss, test_acc = model.evaluate(test_array, test_labels)
# print('Test accuracy:', test_acc)

# model.save('fall_detec_model.h5')

model = keras.models.load_model('fall_detec_model.h5')
input_shape = (1, 128)
model.build(input_shape)
model.summary()
pre_y = model.predict(test_array)
print(test_labels)
print(pre_y)
