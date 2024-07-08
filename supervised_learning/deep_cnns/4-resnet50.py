#!/usr/bin/env python3
"""resnet50 in keras"""
from tensorflow import keras as K


def resnet50():
    """builds resnet50 in keras"""
    return K.applications.resnet50.ResNet50()
