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
            if lambtha < 0:
                return 0
            self.lambtha = float(lambtha)
