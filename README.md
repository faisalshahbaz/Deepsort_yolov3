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

## Overview

Duration：Jun. 2019 - Sep. 2019

Advisor：*Research associate* YUAN Fei

This is my internship project, which aims to detect intrusion, line- crossing, running and falling behaviors automatically in surveillance videos through deep learning algorithms. Through this project, I am trying to reach the following learning objectives:

1. Master the Python language and object-oriented programming (OOP) ideas.
2. Master the TensorFlow deep learning framework.
3. Understand frequently-used target detection algorithms, having implemented traditional algorithms including GMM and ViBe, machine learning algorithm HOG+SVM, and deep learning algorithm YOLOV3. Now the YOLOV3 algorithm for target detection.
4. Understand frequently-used target tracking algorithms. Currently the Deep Sort algorithm for target tracking.
5. Master methods for optimizing computing resources.

This video presents basic functions of this project. 👇

https://www.youtube.com/embed/kFEjHOXokIw


|       Name        |              Configuration              |
| :---------------: | :-------------------------------------: |
|     Processor     | Intel® Core™ i7-8700 CPU @ 3.20GHz × 12 |
|        GPU        |           GeForce GTX 1050 Ti           |
|      Memory       |                  16 GB                  |
| Operating system  |        Ubuntu 16.04.6 LTS 64-bit        |
| Video information |             1280*720 30FPS              |
| Processing speed  |                  7FPS                   |

The code for the YOLOv3 algorithm and the Deep Sort algorithm refer to the following repositories: [Https://github.com/Qidian213/deep_sort_yolov3](Https://github.com/Qidian213/deep_sort_yolov3), [Https://github.com/YunYang1994/tensorflow-yolov3](Https://github.com/YunYang1994/tensorflow-yolov3) and [Https://github.com/nwojke/deep_sort](Https://github.com/nwojke/deep_sort). Owing to these projects! I hope that one day, I could contribute to the open-source community like them.

I tried to train a classifier to identify the falling behaviors based on target detection. However, my samples gathered from YouTube resulted in poor generalization performance.

P.S. I found that YunYang posted a code analysis of YOLOV3. If you are interested, you can go to: [https://github.com/YunYang1994/CodeFun/blob/master/005-paper_reading/YOLOv3.md](https://github.com/YunYang1994/CodeFun/blob/master/005-paper_reading/YOLOv3.md)

## Quick start

  1. Clone this repository.
  2. Download yolov3_coco.pb from https://github.com/YunYang1994/tensorflow-yolov3
  3. Download mars-small128.pb from https://github.com/Qidian213/deep_sort_yolov3.
  4. Run demo.py.

---
## Functions

This part is to explain the meaning of the UI interface.

<img src="https://s2.ax1x.com/2019/10/07/u2Ezb8.png" alt="u2Ezb8.png" border="0" height="500"/>

**General functions:**

*The file path of opening*: Enter the path of a source video, which defaults to "/home/tom/桌面/行人检测算法/测试数据/监控视频/003.avi"

*Save videos*: Decide whether to save the alarm picture. If a target alarms multiple times, save only the first picture.

*The file path of saving*: Enter the saved path of the alarm pictures, which defaults to "./alarm_frame/"

------

**Detecting  intrusion**：

*Does it have a warning area*: Implement detecting intrusion function, determine whether there is an alert area? If yes, a quadrilateral area needs to be defined at four points in the window. When the target enters this area, the box on the target will turn red to alert.

------

**Detecting crossing a line**：

*Does it have a warning line*: Whether there is a warning line? If yes, a straight line needs to be determined in the next picture. When the target crosses this line, the box on the target will turn red to alert. After selecting this option, you need to choose the next two options:

1. *Single cross？* : Is it one-way crossing? The default is two-way crossing, and it will alert as long as a target crossed the line. If you choose one-way crossing, only crossing the line in one direction will alert. After selecting this option, you need to select the next option:

2. *Reverse direction？*: Whether to reverse the alarm direction of one-way crossing.

------

**Detecting running**：

*Does it have a speed limit?*：Implement detecting running function (pixels/frame). If yes, the box on the target will turn red to alert when the speed exceeds the set value. After selecting this option, you need to select the next option:

*Maximum  speed* : Set alarm speed in pixels/frame.

------

**Detecting falling**: At present, this function has a poor effect on the actual video.

*Does It judge fall？*: If yes, implement detecting falling function.

*Falling time*: Enter the minimum speed (pixels/frame). When the target's speed is less than this value, the box on the target will turn red to alert.

