#!/usr/bin/env python3
"""module for RMSProp from scratch"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """updates a given variable using RMSProp"""
    Sderiv = (beta2 * s) + ((1 - beta2) * (grad ** 2))
    mod_var = var - (alpha * (grad / (((Sderiv + epsilon) ** 0.5))))
    return mod_var, Sderiv
