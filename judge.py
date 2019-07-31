"""
Judge if the objects are inside the alarm area
"""

import cv2
import numpy as np


class JUDGE(object):  # Object类是所有类都会继承的类
    def __init__(self, height=1080, width=1920):
        self.black = np.zeros([height, width], dtype=np.uint8)

    # def close(self):
    #     del self.black

    def draw(self, pts, image):
        if pts.size > 4:  # 区域
            cv2.fillPoly(self.black, [pts], (255, 255, 255))
            self.contours, self.hierarchy = cv2.findContours(
                self.black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, self.contours, -1, (0, 0, 255), 2)
        else:
            cv2.line(image, (pts[0, 0], pts[0, 1]), (pts[1, 0], pts[1, 1]),
                     (0, 0, 255), 2)

        return image
