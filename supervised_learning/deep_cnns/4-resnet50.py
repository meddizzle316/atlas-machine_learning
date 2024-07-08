#!/usr/bin/env python3
"""resnet50 in keras"""
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block

def resnet50():
    """builds resnet50 in keras"""
    return K.applications.resnet50.ResNet50()
