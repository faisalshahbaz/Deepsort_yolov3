#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import numpy as np
import core.utils as utils
import tensorflow as tf
from yolov3_tf import YOLOV3

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
# from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
# from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')


def main(yolov3):

    # Definition of the parameters
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

    writeVideo_flag = False

    # video_path = "/home/tom/桌面/行人检测算法/测试视频/test.mp4"
    video_path = "/home/tom/桌面/行人检测算法/people/003.avi"
    video_capture = cv2.VideoCapture(video_path)

    if writeVideo_flag:
        # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('output.avi', fourcc, 15, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1

    fps = 0.0
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        t1 = time.time()

        # image = Image.fromarray(frame)
        # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        boxs = yolov3.detect_image(frame)  # 从这里开始检测
        # boxs = yolov3_tf.detect_image(image)
        # print("box_num",len(boxs))
        features = encoder(frame, boxs)

        # score to 1.0 here).
        detections = [
            Detection(bbox, 1.0, feature)
            for bbox, feature in zip(boxs, features)
        ]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])  # 分数是1
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap,
                                                    scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            # 白色是卡尔曼滤波预测的目标，绿的字
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            # strtemp = str(track.track_id) + " dx = " + str(
            #     round(track.mean[4], 3)) + " dy = " + str(
            #         round(track.mean[5], 3))
            strtemp = str(track.track_id) + " v = " + str(
                round(np.sqrt(track.mean[4]**2 + track.mean[5]**2),
                      3)) + " pixels/frame"
            cv2.putText(frame, strtemp, (int(bbox[0]), int(bbox[1])), 0,
                        5e-3 * 200, (0, 255, 0), 2)

        for det in detections:
            bbox = det.to_tlbr()
            # 蓝色是检测出的目标
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

        cv2.imshow('', frame)

        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index) + ' ')
            if len(boxs) != 0:
                for i in range(0, len(boxs)):
                    list_file.write(
                        str(boxs[i][0]) + ' ' + str(boxs[i][1]) + ' ' +
                        str(boxs[i][2]) + ' ' + str(boxs[i][3]) + ' ')
            list_file.write('\n')

        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %f" % (fps))

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    yolov3.close_session()
    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(YOLOV3())
# if __name__ == '__main__'的意思是：当.py文件被直接运行时，
# if __name__ == '__main__'之下的代码块将被运行；
# 当.py文件以模块形式被导入时，if __name__ == '__main__'之下的代码块不被运行。