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
            # print("this is shape", gridH, " of i", i)
            # print("this is the number of anchors", num_anchors)

            cy = np.tile(np.arange(gh, dtype=np.int32)[:, np.newaxis], [1, gw])
            cx = np.tile(np.arange(gw, dtype=np.int32)[np.newaxis, :], [gh, 1])

            cy = np.repeat(cy[..., np.newaxis], num_anchors, axis=2)
            cx = np.repeat(cx[..., np.newaxis], num_anchors, axis=2)
            # cy = np.expand_dims(cy, -1)
            # cx = np.expand_dims(cx, -1)

            # print("this is the shape of cy", cy.shape)
            # print("this is the shape of cx", cx.shape)
            # y_grid = np.tile(xy_grid[:, :, np.newaxis, :], [1, 1, 3, 1])
            # xy_grid = np.cast(xy_grid, np.float32)

            # pred_x = (self.sigmoid(x) + cx) / gridW
            # pred_y = (self.sigmoid(y) + cy) / gridH
            # pred_w = (np.exp(w) * self.anchors[i, :, 0]) / image_width
            # pred_h = (np.exp(h) * self.anchors[i, :, 1]) / image_height

            pred_x = (self.sigmoid(box[..., 0]) + cx) / gw
            pred_y = (self.sigmoid(box[..., 1]) + cy) / gh
            pred_w = (np.exp(box[..., 2]) * self.anchors[i, :, 0]) / image_w
            pred_h = (np.exp(box[..., 3]) * self.anchors[i, :, 1]) / image_h

            # print("this is the shape of pred x", pred_x.shape)
            #
            # print("this is boxes 0", boxes[0])
            # print("shape of boxes 0", boxes[0].shape)
            boxes[i][..., 0] = (pred_x - (pred_w * 0.5)) * image_w
            boxes[i][..., 1] = (pred_y - (pred_h * 0.5)) * image_h
            boxes[i][..., 2] = (pred_x + (pred_w * 0.5)) * image_w
            boxes[i][..., 3] = (pred_y + (pred_h * 0.5)) * image_h

            # x1 = (pred_x - (pred_w * 0.5)) * image_width
            # y1 = (pred_y - (pred_h * 0.5)) * image_height
            # x0 = (pred_x + (pred_w * 0.5)) * image_width
            # y0 = (pred_y + (pred_h * 0.5)) * image_height

            # print("x1 shape", x1.shape)
            # print("y1 shape", y1.shape)
            # print("x0 shape", x0.shape)
            # print("y0 shape", y0.shape)

            # print("x1 shape", x1.shape)
            # print("y1 shape", y1.shape)

            # box_cord = np.concatenate([x1, y1, x0, y0], axis=-1)
            # box_cord = np.reshape(box_cord, (gridH, gridW, num_anchors, 4))
            # print("shape of box cord", box_cord.shape)
            # boxes.append(box_cord)
            i += 1

        return [boxes, confidence_list, class_probs]
