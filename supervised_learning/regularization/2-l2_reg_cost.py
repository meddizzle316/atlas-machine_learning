#!/usr/bin/env python3
"""l2 regularization cost"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """wow easily the most obfuscated directions yet"""
    return cost + model.losses 



