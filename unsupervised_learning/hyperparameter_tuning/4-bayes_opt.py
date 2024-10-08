#!/usr/bin/env python3
"""
performs bayesian optimization on a noiseless
1d gaussian process
"""
import numpy as np
from scipy.stats import norm
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

    def acquisition(self):
        """does acquisition"""
        mu, sigma_f = self.gp.predict(self.X_s)

        if self.minimize:
            y = np.min(self.gp.Y)
            improvement = (y - mu) - (self.xsi)
        else:
            y = np.max(self.gp.Y)
            improvement = (mu - y) - (self.xsi)

        with np.errstate(divide="warn"):
            Z = improvement / sigma_f
            EI = improvement * norm.cdf(Z) + sigma_f * norm.pdf(Z)
            # handle Zero Uncertainty?
            EI[sigma_f == 0.0] = 0

        next_point = np.argmax(EI)

        n = self.X_s.shape[0]
        return self.X_s[next_point], EI
