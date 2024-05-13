#!/usr/bin/env python3
"""class for normal distribution"""


class Normal:
    """class for normal distribution"""
    def __init__(self, data=None, mean=0, stddev=1.):
        """init func"""
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float((sum(data) / len(data)))
            self.stddev = 0
            temp = 0
            for i in data:
                temp += (i - self.mean) ** 2
            temp = temp / (len(data))
            temp = temp ** 0.5
            self.stddev = temp
        else:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)

    def z_score(self, x):
        """gets z-score of a given x value"""
        return ((x - self.mean) / self.stddev)

    def x_value(self, z):
        """gets z_score of given z value"""
        return (self.stddev * z) + self.mean
