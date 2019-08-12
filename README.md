## Introduction

这是我实习中的项目，目的是通过深度学习算法，自动检测监控视频中的入侵、越线、跑步和跌倒的行为。
通过这个项目，我正在完成以下学习目标：
1. 掌握python语言和面向对象编程思想。
2. 掌握tensorflow深度学习框架。
3. 了解常用的目标检测算法。我已经实现了：传统算法包括GMM和ViBe，机器学习算法HOG+SVM和深度学习算法YOLOV3。目前我使用YOLOV3算法进行目标检测功能。
4. 了解常用的目标追踪算法。目前我使用Deep Sort算法进行目标追踪。
5. 掌握优化计算资源的一些方法。

YOLOv3算法和Deep Sort算法的代码参考了以下几个项目:
 https://github.com/Qidian213/deep_sort_yolov3
 https://github.com/YunYang1994/tensorflow-yolov3
 https://github.com/nwojke/deep_sort
 非常感谢这几个项目！希望有一天我也可以像他们一样为开源社区做贡献。
 
 目前我正在训练一个分类器，尝试在目标检测的基础上，对跌倒，跑步和站立三种行为进行识别。

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

The code for the YOLOv3 algorithm and the Deep Sort algorithm refer to the following repositories:
  https://github.com/Qidian213/deep_sort_yolov3
  https://github.com/YunYang1994/tensorflow-yolov3
  Https://github.com/nwojke/deep_sort
Thanks very much for these projects! I hope that one day I can contribute to the open source community like them.
 
I am currently training a classifier to try to identify the three behaviors of falling, running and standing based on the target detection.

P.S. I saw that YunYang posted a code analysis of YOLOV3. If you are interested, you can move:
https://github.com/YunYang1994/CodeFun/blob/master/005-paper_reading/YOLOv3.md
 
## Quick start
 
 1. Clone this repository.
 
 2. Download yolov3_coco.pb from https://github.com/YunYang1994/tensorflow-yolov3

 3. Download mars-small128.pb from https://github.com/Qidian213/deep_sort_yolov3.

 4. Run demo.py. 现在的代码写的非常差，我会尽快把代码修正得更加清晰易懂。
