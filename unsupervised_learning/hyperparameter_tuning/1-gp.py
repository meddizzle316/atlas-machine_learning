#!/usr/bin/env python3
"""updated Gaussian Process model"""
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

    def predict(self, X_s):
        """predicting std dev and mean"""

        K_s = self.kernel(X_s, self.X)

        mu_s = K_s @ np.linalg.inv(self.K) @ self.Y

        K_ss = self.kernel(X_s, X_s)
        sigma_s = np.diag(K_ss.T - K_s @ np.linalg.inv(self.K) @ K_s.T)

        return mu_s.flatten(), sigma_s.flatten()
