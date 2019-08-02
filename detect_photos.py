#! /usr/bin/env python
# -*- coding: utf-8 -*-

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

for filename in os.listdir(path):
    frame = cv2.imread(path + filename)
    # box 需要 x, y, w, h的格式
    x = 0
    y = 0
    w = frame.shape[0]
    h = frame.shape[1]
    box = [x, y, w, h]

    feature = encoder(frame, [box])
    print(feature)