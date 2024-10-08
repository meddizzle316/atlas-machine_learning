#!/usr/bin/env python3
"""represents a noiseless 1d Gaussian process"""
import numpy as np


class GaussianProcess:
    """a class for a gaussian process"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """inits for gaussian process class"""

        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """initializes covariance matrix with RBF kernel"""
        m, _ = X1.shape
        n, _ = X2.shape
        K = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                K[i, j] = (self.sigma_f ** 2 * np.exp
                           (-(np.linalg.norm
                              (X1[i] - X2[j]) ** 2) / (
                               2 * self.l**2)))
        return K
