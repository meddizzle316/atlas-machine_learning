#!/usr/bin/env python3
"""class for exponential distribution"""


class Exponential:
    """class for exponential distribution"""
    def __init__(self, data=None, lambtha=1.):
        """init funct"""
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float((1 / ((sum(data)) / (len(data)))))
        else:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        self.e = 2.7182818285
        self.pi = 3.1415926536

    def pdf(self, x):
        """gets pdf for given time period"""
        if x < 0:
            return 0
        return (self.lambtha * (self.e ** ((-self.lambtha) * x)))
