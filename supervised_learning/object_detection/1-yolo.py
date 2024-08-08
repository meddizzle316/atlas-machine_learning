#!/usr/bin/env python3
"""makes Yolov3 in Keras"""

import tensorflow as tf
from tensorflow import keras as K
import numpy as np


class Yolo():
    """Yolov3 in keras"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """init yolo"""
        # setting darknet model
        self.model = K.models.load_model(model_path)

        # set class_names
        with open(classes_path, 'r') as file:
            self.class_names = []
            for line in file:
                self.class_names.append(line[:-1])

        # basic instance variables
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """processing outputs into useful formats"""
        boxes = []
        confidences = []
        class_probs = []
        count = 0
        image_height = image_size[0]
        image_width = image_size[1]

        for box in outputs:

            #
            x = box[:, :, :, 0]
            y = box[:, :, :, 1]
            w = box[:, :, :, 2]
            h = box[:, :, :, 3]
            confidences.append(box[:, :, :, 4])
            class_probs.append(box[:, :, :, 5:-1])

            gridH = box.shape[0]
            gridW = box.shape[1]

            cx = gridW * x
            cy = gridH * y
            bx = tf.math.sigmoid(x) + cx
            by = tf.math.sigmoid(y) + cy
            bw = w * image_size
            bh = h * image_size

            x0 = (bx - bw / 2)
            y0 = (by - bh / 2)
            x1 = (bx + bw / 2)
            y1 = (by + bh / 2)

            bb = np.concatenate([x0, y0, x1, y1], axis=-1)
            reshape_bb = bb.reshape(52, 52, 3, -1)
            boxes.append(reshape_bb)
            count += 1
        # return [boxes, confidences, class_probs]