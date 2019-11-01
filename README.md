## Introduction

这是我实习中的项目，目的是通过深度学习算法，自动检测监控视频中的入侵、越线、跑步和跌倒的行为。
通过这个项目，我正在完成以下学习目标：
1. 掌握python语言和面向对象编程思想。
2. 掌握tensorflow深度学习框架。
3. 了解常用的目标检测算法。我已经实现了：传统算法包括GMM和ViBe，机器学习算法HOG+SVM和深度学习算法YOLOV3。目前我使用YOLOV3算法进行目标检测功能。
4. 了解常用的目标追踪算法。目前我使用Deep Sort算法进行目标追踪。
5. 掌握优化计算资源的一些方法。
6. 训练跌倒直立分类器

YOLOv3算法和Deep Sort算法的代码参考了以下几个项目:
 https://github.com/Qidian213/deep_sort_yolov3
 https://github.com/YunYang1994/tensorflow-yolov3
 https://github.com/nwojke/deep_sort
 非常感谢这几个项目！希望有一天我也可以像他们一样为开源社区做贡献。

 之前我训练了一个分类器，尝试在目标检测的基础上，对跌倒进行识别。但从Youtube上收集的跌倒样本训练出的分类器泛化性能太差，因此没有加入。

 P.S. 看到YunYang大神贴出了对YOLOV3的代码剖析，有兴趣的可以移步：
 https://github.com/YunYang1994/CodeFun/blob/master/005-paper_reading/YOLOv3.md

---

This is my internship project, which aims to automatically detect intrusion, cross-over, running and falling behavior in surveillance video through deep learning algorithms.
Through this project, I am trying to complete the following learning objectives:

1. Master Python language and object-oriented programming (OOP) ideas.
2. Master the tensorflow deep learning framework.
3. Understand common target detection algorithms. I have implemented: traditional algorithms include GMM and ViBe, machine learning algorithm HOG+SVM and deep learning algorithm YOLOV3. Currently I use the YOLOV3 algorithm for target detection.
4. Understand common target tracking algorithms. Currently I use the Deep Sort algorithm for target tracking.
5. Master some methods for optimizing computing resources

|       Name        |              Configuration              |
| :---------------: | :-------------------------------------: |
|     Processor     | Intel® Core™ i7-8700 CPU @ 3.20GHz × 12 |
|        GPU        |           GeForce GTX 1050 Ti           |
|      Memory       |                  16 GB                  |
| Operating system  |        Ubuntu 16.04.6 LTS 64-bit        |
| Video information |             1280*720 30FPS              |
| Processing speed  |                  7FPS                   |

The code for the YOLOv3 algorithm and the Deep Sort algorithm refer to the following repositories:
  https://github.com/Qidian213/deep_sort_yolov3
  https://github.com/YunYang1994/tensorflow-yolov3
  Https://github.com/nwojke/deep_sort
Thanks very much for these projects! I hope that one day I can contribute to the open source community like them.

I tried to train a classifier to try to identify the falling behaviors based on target detection. However, my samples gathered from YouTube results in poor generalization performance.

P.S. I saw that YunYang posted a code analysis of YOLOV3. If you are interested, you can move:
https://github.com/YunYang1994/CodeFun/blob/master/005-paper_reading/YOLOv3.md

## Quick start

  1. Clone this repository.
  2. Download yolov3_coco.pb from https://github.com/YunYang1994/tensorflow-yolov3
  3. Download mars-small128.pb from https://github.com/Qidian213/deep_sort_yolov3.
  4. Run demo.py.

---

### 概述

该demo实现了入侵，越线，跌倒和跑步的功能，详见简易需求文档。

主要算法包括：目标检测算法+目标追踪算法+跌倒分类器。

目标检测算法可以判断图片中有哪些物体，每个物体的位置（方框的左上和右下的坐标），实现入侵和越线的功能；目标追踪算法可以通过卡尔曼滤波对每一个目标进行预测与匹配，从速度层面对跑步实行报警，也可以判断多长时间以后开启跌倒检测器；检测器可以判断检测框内部的人是站立还是跌倒。

目标检测算法：YOLOV3

目标追踪算法：Deep Sort

跌倒分类器：我使用Deep Sort自带的Marz网络进行特征提取，提取之后为128维的向量，之后用一个三层全连接网络进行分类。

### 如何使用demo

1. 打开VScode
2. 运行demo.py文件
3. ![1568015336283](/home/tom/.config/Typora/typora-user-images/1568015336283.png)

需要在该界面上进行设置，以启动相应的功能。

------

**整体功能**：

The file path of opening：输入源视频路径，默认为"/home/tom/桌面/行人检测算法/测试数据/监控视频/003.avi"。

Save videos：是否将报警的图片保存。一个目标如果多次报警，只保存第一张图片。

The file path of saving：输入报警的图片保存的路径。默认为"./alarm_frame/"

------

**入侵功能**：

Does it have a warning area：实现入侵功能，是否有警戒区域。如果有，在下一幅画面上需要点四个位置划定一个四边形区域，当目标进入该区域会变红报警。

------

**越界功能**：

Does it have a warning line：实现越线功能，是否有警戒线。如果有，在下一幅画面上需要点两下确定一条直线，当目标跨过这个区域会变红报警。在选了这个选项的基础上需要选择下个选项：

​			Single cross？ ：是否是单向穿越。默认是双向穿越，目标过线就会报警。如果选择单向穿越，只有从某一方向过线才会报警。在选了这个的基础上需要选择下个选项：

​			Reverse direction？：是否反转单向穿越的报警方向。

------

**检测速度功能**：

Does it have a speed limit?：实现检测速度功能（单位是像素/帧）。如果有，当速度超过设定值，目标会变红。在选了这个的基础上需要下个选项：

​	Maximum  speed：设定报警速度，单位是像素/帧

------

**检测跌倒功能**：当前这个功能对实际视频效果很差

Does it judge fall？：勾选则实现对跌倒的检测，跌倒会报警。

Fall time：输入设定的最小速度，单位是像素/帧。当目标的速度小于这个值，会启用检测器进行跌倒检测。

如果有问题，可以邮件联系我 lijinjie362@outlook.com
