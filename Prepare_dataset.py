#! /usr/bin/env python
# -*- coding: utf-8 -*-
# 这段代码可以将图片文件变成TFRecord数据集，之后再变成dataset数据集
import os
import tensorflow as tf
# from PIL import Image
from sklearn.utils import shuffle
import numpy as np
from tqdm import tqdm
from core import utils
import cv2
import math
import random


def load_sameple(sample_dir, shuffleflag=True):
    # 递归读取文件。只支持一级。返回文件名、数值标签、数值对应的标签名
    print('loading sample dataset...')
    lfilenames = []
    labelsnames = []
    for (dirpath, dirnames, filenames) in os.walk(sample_dir):
        # 递归遍历文件夹
        for filename in filenames:  # 遍历所有文件名
            filename_path = os.sep.join([dirpath, filename])
            lfilenames.append(filename_path)  # 添加文件名
            labelsnames.append(dirpath.split('\\')[-1])  # 添加文件名对应的标签

    lab = list(sorted(set(labelsnames)))  # 生成标签名称列表
    labdict = dict(zip(lab,
                       list(range(len(lab)))))  # 生成字典；zip()将迭代器的元素都生成然后打包传回

    labels = [labdict[i] for i in labelsnames]
    if shuffleflag is True:
        return shuffle(np.asarray(lfilenames),
                       np.asarray(labels)), np.asarray(lab)
    else:
        return (np.asarray(lfilenames), np.asarray(labels)), np.asarray(lab)


def makeTFRec(filenames, labels):  # 定义生成TFRecord的函数
    # 定义 writer, 用于向TFRecords文件写入数据
    input_size = 416
    writer = tf.python_io.TFRecordWriter("mydata.tfrecords")
    for i in tqdm(range(0, len(labels))):  # 调用进度条显示
        img = cv2.imread(filenames[i])
        img = utils.image_preporcess(img, [input_size, input_size])
        cv2.imshow('', img)
        cv2.waitKey(0)
        img_raw = img.tobytes()  # 将图片转化为二进制格式
        example = tf.train.Example(features=tf.train.Features(
            feature={
                "label":  # 存放图片的标签label
                tf.train.Feature(int64_list=tf.train.Int64List(value=[
                    labels[i]
                ])),
                "img_raw":  # 存放具体的图片 img_raw
                tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))  # 用example对象对label和image数据进行封装

        writer.write(example.SerializeToString())  # 序列转化为字符串
    writer.close()


def main():
    # =======================制作TFRecord数据===================
    # directory = '/home/tom/桌面/行人检测算法/Fall_Upright_people'
    # (filenames, labels), _ = load_sameple(directory, shuffleflag=False)
    # makeTFRec(filenames, labels)
    # Definition of the parameters
    # =======================================================

    path_fall = "/home/tom/桌面/行人检测算法/people/Fall_Augmentation/"
    path_upright = "/home/tom/桌面/行人检测算法/people/Upright_Augmentation/"

    # ========================翻转样本==========================
    # for filename in os.listdir(path_fall):
    #     frame = cv2.imread(path_fall + filename)
    #     frame_horizontal = frame.copy()
    #     frame_horizontal = cv2.flip(frame, 1)
    #     filename = os.path.splitext(filename)[0]
    #     cv2.imwrite("%s%s.png" % (path_fall, filename + '_r'),
    #                 frame_horizontal)
    # ===========================================================

    # =======================随机截取样本========================
    # for filename in os.listdir(path_fall):
    #     frame = cv2.imread(path_fall + filename)
    #     frame_interception = frame.copy()
    #     size = frame_interception.shape
    #     h = math.floor(size[0] * 0.8)  # 被截取图像的高和宽
    #     w = math.floor(size[1] * 0.8)
    #     y = random.randint(0, math.floor(size[0] * 0.2))
    #     x = random.randint(0, math.floor(size[1] * 0.2))

    #     frame_interception = frame[y:h, x:w, :]

    #     filename = os.path.splitext(filename)[0]
    #     cv2.imwrite("%s%s.png" % (path_fall, filename + '_i'),
    #                 frame_interception)
    # ======================================================

    # =================== 改变光照等条件 ========================
    path = path_upright
    num = 0
    for filename in os.listdir(path):
        frame = cv2.imread(path + filename)

        # alpha * src + beta
        # alpha = 0.1 * random.randrange(10, 30, 1)
        alpha = 0.1 * random.randrange(10, 13, 1)
        beta = random.randrange(-30, 30, 1)

        frame_brightness = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        filename = os.path.splitext(filename)[0]
        cv2.imwrite("%s%s.png" % (path, filename + '_b'), frame_brightness)
        num += 1
        print(num)


if __name__ == '__main__':
    main()