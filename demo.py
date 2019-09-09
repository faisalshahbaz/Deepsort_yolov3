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
# from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')

# /home/tom/桌面/行人检测算法/测试视频/Funniest Falls.mp4

tag = -1
ix = [0, 0, 0, 0]
iy = [0, 0, 0, 0]
drawing = False


# 鼠标出现动作后调用的函数
def draw_area(event, x, y, flags, param):
    global ix, iy, drawing, tag
    #当按下左键时返回起始位置坐标
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        tag = tag + 1
        if tag < 4:
            ix[tag], iy[tag] = x, y


def draw_line(event, x, y, flags, param):
    global ix, iy, drawing, tag
    #当按下左键时返回起始位置坐标
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        tag = tag + 1
        if tag < 2:
            ix[tag], iy[tag] = x, y


def main(yolov3):
    global ix, iy, drawing, tag, img
    order = 0  # 统计是第几帧
    alarm_tag = False  # 这一帧是否报警
    person_list = []  # 储存person_ID的list

    G = Gui()
    G.gui()

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

    # video_path = "/home/tom/桌面/行人检测算法/测试视频/test.mp4"
    # video_path = "/home/tom/桌面/行人检测算法/people/003.avi"
    video_path = G.pathToLoad

    video_capture = cv2.VideoCapture(video_path)

    # ================= 储存视频 =================
    # if G.ifsave == 1:
    #
    # ==============获取鼠标事件画区域的代码=================
    if G.ifregion == 1:  # 如果画警戒区域
        value, img = video_capture.read()
        # rotate the img
        # img = np.rot90(img, -1)

        cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(
            "image", draw_area)  # 第二个参数为回调函数，即指定窗口里每次鼠标事件发生的时候被调用的函数指针。
        while (1):
            if drawing is True and tag < 4:
                if tag > 0:
                    cv2.line(img, (ix[tag - 1], iy[tag - 1]),
                             (ix[tag], iy[tag]), (0, 0, 255), 2)
                if tag == 3:
                    cv2.line(img, (ix[0], iy[0]), (ix[tag], iy[tag]),
                             (0, 0, 255), 2)
                drawing = False

            cv2.imshow('image', img)
            k = cv2.waitKey(1)
            if k == ord('q') or tag == 4:
                break
        pts = np.array([[ix[0], iy[0]], [ix[1], iy[1]], [ix[2], iy[2]],
                        [ix[3], iy[3]]])
        cv2.destroyWindow("image")
        # ==============获取鼠标事件画警戒线的代码=================
    if G.ifline == 1:  # 如果画警戒线
        value, img = video_capture.read()
        # rotate the img
        # img = np.rot90(img, -1)
        cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(
            "image", draw_line)  # 第二个参数为回调函数，即指定窗口里每次鼠标事件发生的时候被调用的函数指针。
        while (1):
            if drawing is True and tag < 2:
                if tag > 0:
                    cv2.line(img, (ix[tag - 1], iy[tag - 1]),
                             (ix[tag], iy[tag]), (0, 0, 255), 2)
                drawing = False

            cv2.imshow('image', img)
            k = cv2.waitKey(1)
            if k == ord('q') or tag == 2:
                break
        pts = np.array([[ix[0], iy[0]], [ix[1], iy[1]]])
        cv2.destroyWindow("image")
    # ===============================

    # ============读取=============
    if not ('pts' in dir()):
        pts = np.array([])
    judge = JUDGE(pts)
    fps = 0.0
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        order = order + 1
        if ret is not True:
            break
        t1 = time.time()

        # image = Image.fromarray(frame)
        # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # image = Image.fromarray(frame[..., ::-1])  # bgr to rgb

        # ============检测=============
        bboxes = yolov3.detect_image(frame)  # 从这里开始检测

        # =============测试Bbox的坐标对不对=============
        # for bbox in bboxes:
        #     x1 = bbox[0]
        #     y1 = bbox[1]
        #     x2 = bbox[0] + bbox[2] - 1
        #     y2 = bbox[1] + bbox[3] - 1
        #     # photo1 = frame[int(bbox[0]):int(bbox[2]),
        #     #                int(bbox[1]):int(bbox[3])]
        #     # cv2.imshow('photo1', photo1)
        #     # cv2.waitKey(0)
        #     photo2 = frame[int(y1):int(y2 + 1), int(x1):int(x2 + 1), :]
        #     cv2.imshow('photo1', photo2)
        #     cv2.waitKey(0)
        #     # 如果Bbox的坐标是正确形式，那么photo2是正确的，photo1是错误的
        # ============================================

        features = encoder(frame, bboxes)  # boxes必须是x, y, w, h的格式

        # score to 1.0 here).
        detections = [
            Detection(bbox, 1.0, feature)
            for bbox, feature in zip(bboxes, features)
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
            vx = track.mean[4]
            vy = track.mean[5]
            v = np.sqrt(vx**2 + vy**2)
            v = judge.filter_vel(track.track_id, v)  # 对速度进行平滑处理
            bbox = track.to_tlbr()  # 这个已经从xywh转成xyxy格式了
            color = (255, 255, 255)  # default color

            # 如果有限制速度
            if G.ifspeed == 1:
                if v > G.speedMax:
                    if track.track_id not in person_list:
                        person_list.append(track.track_id)
                        alarm_tag = True  # alarm_tag 仅用于指示保存
                    color = (0, 0, 255)

            # 如果有跌倒检测
            if G.iffall == 1:
                # if not track.is_confirmed():
                #     continue
                sample = np.array(
                    track.features_cons[0:1]
                )  # features_cons 里边保存了多次的检测数据，以应对遮挡等突然变化的情况. 原为[-1:0]

                # if v < G.speedMin and sample.size == 128:
                # if sample.size == 128:
                #     result = judge.model.predict(sample)
                #     if len(result) != 0:
                #         result = np.array(result[0])
                #         if ((result[1] - result[0]) > 0.5):  # 第二个数字大是跌倒
                #             color = (0, 0, 255)
                #             print(result)

                if track.track_id:
                    if judge.determine_falling(track.track_id, sample) is True:
                        color = (0, 0, 255)

            # 如果中心点或底边中点落入警戒区域，则变红。警戒才有这一部分。
            if judge.determine(
                (bbox[0] + bbox[2]) / 2, bbox[3]) or judge.determine(
                    (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2):

                if judge.determine_direction(vx, vy, G.ifsingle_cross,
                                             G.ifregion, G.ifreverse):
                    if track.track_id not in person_list:
                        person_list.append(track.track_id)
                        alarm_tag = True  # alarm_tag 仅用于指示保存
                    color = (0, 0, 255)

            # ============积累样本搞的临时代码==============
            # if 'people_number' not in dir():
            #     people_number = 0

            # if alarm_tag is True:
            #     people_number += 1
            #     photo = frame[int(bbox[0]):int(bbox[2]),
            #                   int(bbox[1]):int(bbox[3])]
            #     cv2.imshow('abc', photo)
            # ===========================================

            # 白色是卡尔曼滤波预测的目标，绿的字
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])), color, 2)

            strtemp = str(track.track_id) + " v = " + str(round(
                v, 3)) + " pixels/frame"
            cv2.putText(frame, strtemp, (int(bbox[0]), int(bbox[1])), 0,
                        5e-3 * 200, (0, 255, 0), 2)

        for det in detections:
            bbox = det.to_tlbr()
            # 蓝色是检测出的目标
            color = (255, 0, 0)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])), color, 2)

        # ==============绘制警戒区+作图==================
        frame = judge.draw(frame)
        cv2.imshow('', frame)
        if (alarm_tag is True) and (G.ifsave == 1):
            cv2.imwrite("%s%s.jpg" % (G.pathToSave, order), frame)
            alarm_tag = False
        # ==============储存视频 =====================
        # if G.ifsave:
        # ============================================

        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %f" % (fps))

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ========= 结束全流程 ========
    yolov3.close_session()
    video_capture.release()
    # if G.ifsave == 1:
    #     out.release()
    #     list_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(YOLOV3())
# if __name__ == '__main__'的意思是：当.py文件被直接运行时，
# if __name__ == '__main__'之下的代码块将被运行；
# 当.py文件以模块形式被导入时，if __name__ == '__main__'之下的代码块不被运行。