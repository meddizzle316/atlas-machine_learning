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

    def filter_boxes(self, boxes_list, box_conf, class_probs):
        """filtering box inputs from preprocess inputs"""

        above_t = None
        box_scores = None
        class_prob = None

        box_score = [np.multiply(c, p) for c, p in zip(box_conf, class_probs)]
        box_class = [np.argmax(score, axis=-1) for score in box_score]
        box_score = [np.amax(score, axis=-1) for score in box_score]
        mask = [score >= self.class_t for score in box_score]

        for x, (box, mask) in enumerate(zip(boxes_list, mask)):
            if x > 0:
                above_t = np.concatenate((above_t, box[mask]), axis=0)
                box_scores = np.concatenate((box_scores, box_score[x][mask]))
                class_prob = np.concatenate((class_prob, box_class[x][mask]))
            else:
                above_t = box[mask]
                box_scores = box_score[x][mask]
                class_prob = box_class[x][mask]

        return above_t, class_prob, box_scores

    def box_area(self, box):
        """cals box area"""
        return (box[2] - box[0]) * (box[3] - box[1])

    def box_iou_batch(self, boxes_a, boxes_b):
        """does Iou calc"""
        area_a = self.box_area(boxes_a.T)
        area_b = self.box_area(boxes_b.T)

        top_left = np.maximum(boxes_a[:, None, :2], boxes_b[:, :2])
        bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[:, 2:])

        area_inter = np.prod(
            np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)

        return area_inter / (area_a[:, None] + area_b - area_inter)

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """does non max suppression on boxes with scores >= self.class_t"""

        iou_thresh = self.nms_t
        box_predictions, predicted_box_classes = None, None
        predicted_box_scores = None

        keep = None
        new_classes = []
        new_scores = None
        new_boxes = None
        for cls in np.unique(box_classes):
            class_mask = tf.equal(box_classes, cls)
            class_boxes = tf.boolean_mask(filtered_boxes, class_mask).numpy()
            class_scores = tf.boolean_mask(box_scores, class_mask).numpy()

            if class_boxes.shape[0] > 0:
                indices = tf.image.non_max_suppression(tf.cast(class_boxes, np.float32), tf.cast(
                    class_scores, np.float32), filtered_boxes.shape[0], iou_threshold=iou_thresh)
                if new_scores is None and new_boxes is None:
                    new_scores = class_scores[tf.cast(indices, tf.int32)]
                    new_boxes = class_boxes[tf.cast(indices, tf.int32)]
                else:
                    new_scores = tf.concat(
                        (new_scores, class_scores[indices]), axis=0)
                    new_boxes = tf.concat(
                        (new_boxes, class_boxes[indices]), axis=0)
                for x in range(len(indices)):
                    new_classes.append(cls)

        return new_boxes.numpy(), np.array(new_classes), new_scores.numpy()

        # non_max_idxs = tf.image.non_max_suppression(filtered_boxes, box_scores, filtered_boxes.shape[0], iou_thresh)
        #
        # run = tf.keras.backend.eval
        # new_boxes = run(tf.cast(tf.gather(filtered_boxes, non_max_idxs), tf.int32))
        # new_scores = run(tf.gather(box_scores, non_max_idxs))
        # new_classes = run(tf.gather(box_classes, non_max_idxs))
        #
        # idx = np.argsort(new_classes)
        # ord_class = new_classes[idx]
        # ord_score = new_scores[idx]
        # ord_boxes = filtered_boxes[idx]
        #
        # return ord_boxes, ord_class, ord_score
