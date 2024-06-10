#!/usr/bin/env python3
"""module for momentum"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """updates a specific variable when given the gradient"""
    Vderiv = (beta1 * v) + ((1 - beta1) * grad)
    mod_var = var - (alpha * Vderiv)
    return mod_var, Vderiv
