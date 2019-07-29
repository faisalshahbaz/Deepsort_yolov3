"""
Run a tensorflow YOLO_v3 style detection model.
"""

import cv2
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image

return_elements = [
    "input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0",
    "pred_lbbox/concat_2:0"
]
pb_file = "./yolov3_coco.pb"
# image_path = "./docs/images/road.jpeg"
num_classes = 80
input_size = 416
graph = tf.Graph()


class YOLOV3(object):  # Object类是所有类都会继承的类
    def __init__(self):
        self.return_elements = [
            "input/input_data:0", "pred_sbbox/concat_2:0",
            "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"
        ]
        self.pb_file = "./yolov3_coco.pb"
        self.num_classes = 80
        self.input_size = 416
        self.graph = tf.Graph()
        self.return_tensors = utils.read_pb_return_tensors(
            graph, pb_file, return_elements)
        self.sess = tf.Session(graph=graph)

    def close_session(self):
        self.sess.close()

    def detect_image(self, original_image):

        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image_size = original_image.shape[:2]
        image_data = utils.image_preporcess(np.copy(original_image),
                                            [input_size, input_size])
        image_data = image_data[np.newaxis, ...]

        pred_sbbox, pred_mbbox, pred_lbbox = self.sess.run(
            [
                self.return_tensors[1], self.return_tensors[2],
                self.return_tensors[3]
            ],
            feed_dict={self.return_tensors[0]: image_data})

        pred_bbox = np.concatenate([
            np.reshape(pred_sbbox, (-1, 5 + num_classes)),
            np.reshape(pred_mbbox, (-1, 5 + num_classes)),
            np.reshape(pred_lbbox, (-1, 5 + num_classes))
        ],
                                   axis=0)

        bboxes = utils.postprocess_boxes(pred_bbox, original_image_size,
                                         input_size, 0.3)
        return bboxes
        # bboxes = utils.nms(bboxes, 0.45, method='nms')
        # image = utils.draw_bbox(original_image, bboxes)
        # image = Image.fromarray(image)
        # image.show()
