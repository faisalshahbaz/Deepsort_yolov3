# ! /usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras


class JUDGE(object):  # Object类是所有类都会继承的类
    def __init__(self, pts, height=1080, width=1920):
        self.black = np.zeros([height, width], dtype=np.uint8)
        self.pts = pts
        self.fall_list_id = dict()  # dict
        self.list_id = []
        self.list_v = []
        self.model = keras.models.load_model('fall_detec_model_09830.h5')
        input_shape = (1, 128)
        self.model.build(input_shape)
        self.model.summary()
        # 目前还用不到摔倒检测。其实摔倒检测这个应该单独封装一个类的

    # def close(self):
    #     del self.black

    def draw(self, image):
        # 作用：在图像上绘制警戒区域
        # Input: image
        # Return: image with alarm area or alarm line
        if (self.pts.size < 2):
            return image
        elif self.pts.size > 4:  # 区域
            cv2.fillPoly(self.black, [self.pts], (255, 255, 255))
            self.contours, self.hierarchy = cv2.findContours(
                self.black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, self.contours, -1, (0, 0, 255), 2)
        else:
            cv2.line(image, (self.pts[0, 0], self.pts[0, 1]),
                     (self.pts[1, 0], self.pts[1, 1]), (0, 0, 255), 2)
        return image

    def determine(self, x, y):
        # 作用：判断是否满足警戒条件, inside alarm area of cross alarm line
        # Input：
        # @ x：X coordinates
        # @ y: Y coordinates
        # Return: True/ False
        if (self.pts.size < 2):
            return False
        elif (self.pts.size > 4):  # 区域
            r1 = (self.pts[1, 1] -
                  self.pts[0, 1]) / (self.pts[1, 0] - self.pts[0, 0]) * (
                      x - self.pts[0, 0]) + self.pts[0, 1] - y
            r2 = (self.pts[2, 1] -
                  self.pts[3, 1]) / (self.pts[2, 0] - self.pts[3, 0]) * (
                      x - self.pts[3, 0]) + self.pts[3, 1] - y
            r3 = (self.pts[2, 1] -
                  self.pts[1, 1]) / (self.pts[2, 0] - self.pts[1, 0]) * (
                      x - self.pts[1, 0]) + self.pts[1, 1] - y
            r4 = (self.pts[3, 1] -
                  self.pts[0, 1]) / (self.pts[3, 0] - self.pts[0, 0]) * (
                      x - self.pts[0, 0]) + self.pts[0, 1] - y
            if (r1 * r2 < 0) and (r3 * r4 < 0):
                return True
            else:
                return False
        else:
            # 到警戒线的距离在一定范围以内，防止出现跳跃的情形
            if (abs((self.pts[1, 1] - self.pts[0, 1]) /
                    (self.pts[1, 0] - self.pts[0, 0]) *
                    (x - self.pts[0, 0]) + self.pts[0, 1] - y) < 20):
                return True
            else:
                return False

    def determine_direction(self, vx, vy, ifsingle_cross, ifregion, ifreverse):
        # 作用：判断是否满足单向穿越条件；如果是双向穿越，则一直返回True
        # Input：
        # @ vx：velocty of x direction
        # @ vy: velocity of y direction
        # @ ifsingle_cross : yes, value = 1; no, value = 0
        # @ Return: Bool value
        # 法向量normal vector: (k,-1)
        if (ifregion == 1) or (ifsingle_cross == 0):
            return True  # 如果是警戒区域，或者没有开单向穿越功能
        else:
            self.normal_vec = np.array([(self.pts[1, 1] - self.pts[0, 1]) /
                                        (self.pts[1, 0] - self.pts[0, 0]), -1])
            velocity = np.array([vx, vy])
            if ifreverse == 0:
                if np.dot(self.normal_vec, velocity) > 0:
                    return True
            else:
                if np.dot(self.normal_vec, velocity) < 0:
                    return True
            return False

    def filter_vel(self, ID, v):
        # 作用：对某个对象进行滤波
        # Input：
        # @ id：the id of the object
        # @ v: velocity of the object
        # @ Return: velocity after filtering
        # first time
        if ID not in self.list_id:
            if len(self.list_id) > 100:  # 存储速度数据的上限是100人
                del (self.list_id[0])
                del (self.list_v[0])
            self.list_id.append(ID)
            self.list_v.append([v, 0, 0, 0, 0])
            return v / 5
        else:
            i = self.list_id.index(ID)
            v_array = self.list_v[i]
            v_array[1:] = v_array[0:-1]
            v_array[0] = v
            self.list_v[i] = v_array

            # 中值滤波
            array = sorted(v_array)
            half = len(array) // 2
            return (array[half] + array[~half]) / 2
            # 均值滤波
            # return sum(v_array) / len(v_array)

    def determine_falling(self, ID, Sample, n_frames):
        # 作用：判断某个对象是否已经满足跌倒判定条件。
        # Input:
        # @ ID: the id of the object which needs to judge falling
        # @ Sample: the feature of the object
        # @ Return Bool   T/F
        removed_th = 5  # 如果有removed_th帧这个目标没有跌倒，则从字典中剔除这个ID
        fall_th = n_frames  # 如果有fall_th帧这个目标都跌倒了，则判断为跌倒
        if Sample.size == 128:
            result = self.model.predict(Sample)
            if len(result) != 0:
                result = np.array(result[0])
                if ((result[1] - result[0]) > 0.5):  # 满足跌倒判据
                    if self.fall_list_id.__contains__(ID):  # 如果有这个ID
                        self.fall_list_id[ID] += 2
                    else:
                        self.fall_list_id[ID] = removed_th  # 没有这个ID，加一个
                else:
                    return False
        else:
            return False

        if bool(self.fall_list_id):
            for k in self.fall_list_id:
                self.fall_list_id[k] -= 1  # 所有值自动减一
            for k in self.fall_list_id.copy():  # 必须用copy迭代，因为dict在迭代的时候不能改变
                if self.fall_list_id.get(k) == 0:
                    self.fall_list_id.pop(k)  # 如果太长时间不跌倒，就移除

        if self.fall_list_id.get(ID) >= (removed_th + fall_th):
            self.fall_list_id[ID] += 2  # 一旦判定为真正跌倒，就加一些生命值2，得以保持跌倒状态
            return True
        else:
            return False
