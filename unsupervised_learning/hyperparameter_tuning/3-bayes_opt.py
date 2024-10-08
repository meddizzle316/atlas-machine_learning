#!/usr/bin/env python3
"""
performs bayesian optimization on a noiseless
1d gaussian process
"""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """class for bayesian opt operation"""

    def __init__(
            self,
            f,
            X_init,
            Y_init,
            bounds,
            ac_samples,
            l=1,
            sigma_f=1,
            xsi=0.01,
            minimize=True):
        """init operation"""
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.xsi = xsi
        self.minimize = minimize
        min, max = bounds
        self.X_s = np.linspace(min, max, ac_samples).reshape(-1, 1)
