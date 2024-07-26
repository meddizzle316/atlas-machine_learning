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
