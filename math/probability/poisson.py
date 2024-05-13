#!/usr/bin/env python3
"""Attempting a Poisson Distribution"""


class Poisson:
    """Poisson Class"""
    def __init__(self, data=None, lambtha=1.):
        self.data = data
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            total_events = sum(data)
            lambtha_value = total_events / len(data)
            self.lambtha = float(lambtha_value)
        else:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        self.e = 2.7182818285
        self.pi = 3.1415926536

    def pmf(s, k):
        """gets the probability mass function"""
        k = int(k)
        if k < 0:
            return 0
        return (((s.e ** (-s.lambtha)) * (s.lambtha ** k)) / s.factorial(k))

    def cdf(self, k):
        if k < 0:
            return 0
        k = int(k)
        cdf = 0
        for k in range(k + 1):
            cdf += self.pmf(k)
        return cdf

    def factorial(self, n):
        """gets the factorial of a number"""
        if n == 0 or n == 1:
            return 1
        else:
            return n * self.factorial(n - 1)
