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
        self.e = 2.7182818285
        self.pi = 3.1415926536

    def z_score(self, x):
        """gets z-score of a given x value"""
        return ((x - self.mean) / self.stddev)

    def x_value(self, z):
        """gets z_score of given z value"""
        return (self.stddev * z) + self.mean

    def pdf(s, x):
        """gets pdf for given x"""
        first_factor = 1 / (((s.stddev ** 2) * (2 * s.pi)) ** 0.5)
        second_factor = s.e ** ((-(x - s.mean) ** 2) / ((2 * (s.stddev ** 2))))
        return (first_factor * second_factor)
    
    def erf(self, x):
        """a error function"""
        final_product = 2 / (self.pi ** 0.5)
        second_product = (x - (1/3 * (x**3)) + (1/10 * (x**5)) - (1/42 * (x**7)) + (1/216 * (x ** 9)) )
        return (final_product * second_product)

    def cdf(self, x):
        """Cumulative distribution function for standard normal."""
        z = (self.mean - x) / (self.stddev * (2.0 ** 0.5))
        return 0.5 * ((1 - self.erf(z)))
