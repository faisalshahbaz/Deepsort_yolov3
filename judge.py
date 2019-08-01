# ! /usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np


class JUDGE(object):  # Object类是所有类都会继承的类
    def __init__(self, pts, height=1080, width=1920):
        self.black = np.zeros([height, width], dtype=np.uint8)
        self.pts = pts

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

    def determine_single(self, vx, vy, ifsingle_cross, ifregion, ifreverse):
        # 作用：判断是否满足单向穿越条件；如果是双向穿越，则一直返回True
        # Input：
        # @ vx：velocty of x direction
        # @ vy: velocity of y direction
        # @ ifsingle_cross : yes, value = 1; no, value = 0
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
