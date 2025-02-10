#!/usr/bin/env python3
"""crops an image in tensorflow"""
import tensorflow as tf


def crop_image(image, size):
    """crops image in tf"""
    return tf.image.crop_and_resize(image, crop_size=size)