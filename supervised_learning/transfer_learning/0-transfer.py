#!/usr/bin/env python3
"""preprocessing """
from tensorflow import keras as K


def preprocess_data(X, Y):
  """preprocessing for ResNet50"""
  X = K.applications.resnet50.preprocess_input(X)
  Y = K.utils.to_categorical(Y)
  return X, Y
