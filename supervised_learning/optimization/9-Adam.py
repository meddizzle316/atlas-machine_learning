#!/usr/bin/env python3
"""adam alogrithm"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """adam algorithm from scratch"""
    Vderiv1 = (beta1 * v) + ((1 - beta1) * grad)
    Sderiv = (beta2 * s) + ((1 - beta2) * (grad ** 2))
    Vdcorrected = Vderiv1 / (1 - (beta1 ** t))
    Sdcorrected = Sderiv / (1 - (beta2 ** t))
    mod_var = var - (alpha * (Vdcorrected / ((Sdcorrected + epsilon) ** 0.5)))
    return mod_var, Vderiv1, Sderiv
