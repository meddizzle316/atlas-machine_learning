#!/usr/bin/env python3
"""learning rate decay"""
import numpy as np


def learning_rate_decay(alpha_init, decay_rate, global_step, decay_step):
    """learning rate decay"""
    return (1 / (1 + decay_rate * (global_step // decay_step))) * alpha_init
