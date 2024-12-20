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
        self.strides = [32, 16, 8]

    def sigmoid(self, x):
        """sigmoid function to not use tf.sigmoid"""
        return 1 / (1 + np.exp(-1 * x))

    def process_outputs(self, outputs, image_size):
        """processing outputs into useful formats"""
        boxes = [output[..., :4] for output in outputs]
        # boxes = []
        confidence_list = []
        class_probs = []
        image_h = image_size[0]
        image_w = image_size[1]
        input_layer = self.model.input
        input_w, input_h = input_layer.shape[1:3]

        i = 0
        for output in outputs:

            box = output
            x = box[..., 0]
            y = box[..., 1]
            w = box[..., 2]
            h = box[..., 3]
            confidence = box[:, :, :, 4:5]
            # print("this is confidences", confidence)
            sig_conf = self.sigmoid(confidence)
            # print("this is sig confidence", sig_conf)
            confidence_list.append(sig_conf)
            class_probs.append(self.sigmoid(box[:, :, :, 5:]))

            gh = box.shape[0]
            gw = box.shape[1]
            num_anchors = output.shape[2]

            cx = np.arange(gw).reshape(1, gw)
            cx = np.repeat(cx, gh, axis=0)
            cy = np.arange(gw).reshape(1, gw)
            cy = np.repeat(cy, gh, axis=0).T

            cy = np.repeat(cy[..., np.newaxis], num_anchors, axis=2)
            cx = np.repeat(cx[..., np.newaxis], num_anchors, axis=2)

            pred_x = (self.sigmoid(box[..., 0]) + cx) / gw
            pred_y = (self.sigmoid(box[..., 1]) + cy) / gh
            # print("this is the anchor", self.anchors[i, :, 0], "at ", i)
            pred_w = (np.exp(box[..., 2]) * self.anchors[i, :, 0]) / input_w
            pred_h = (np.exp(box[..., 3]) * self.anchors[i, :, 1]) / input_h
            # input w and input h should be 416
            # they are the 'inputs' to the Yolo model itself
            # the input layer of the yolov3 model
            # is 416, 416
            # this is not the image_size

            boxes[i][..., 0] = (pred_x - (pred_w * 0.5)) * image_w
            boxes[i][..., 1] = (pred_y - (pred_h * 0.5)) * image_h
            boxes[i][..., 2] = (pred_x + (pred_w * 0.5)) * image_w
            boxes[i][..., 3] = (pred_y + (pred_h * 0.5)) * image_h

            i += 1

        return [boxes, confidence_list, class_probs]
