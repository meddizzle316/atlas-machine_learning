#!/usr/bin/env python3
"""a class for binomial distribution"""


class Binomial:
    """class for binomial distribution"""
    def __init__(self, data=None, n=1, p=0.5):
        """init"""
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)

            # took standard deviation from normal class
            self.stddev = 0
            temp = 0
            for i in data:
                temp += (i - self.mean) ** 2
            temp = temp / (len(data))
            temp = temp ** 0.5
            self.stddev = temp
            # getting variance from stddev
            self.var = self.stddev ** 2

            # getting p from inverted m = np formula
            self.p = (self.var - self.mean) / -(self.mean)

            # getting n from inverted m = np formula
            self.n = round(self.mean / self.p)

            # recalculating p
            self.p = float(self.mean / self.n)
        else:
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            if n <= 0:
                raise ValueError("n must be a positive value")
            self.p = float(p)
            self.n = int(n)
        self.e = 2.7182818285
        self.pi = 3.1415926536

    def f(self, n):
        """gets the factorial of a number"""
        result = 1
        int_n = int(n)
        for i in range(1, int_n + 1):
            result *= i
        return result

    def pmf(self, k):
        """probability mass function of binomial"""
        if k < 0:
            return 0
        k = int(k)
        first_factor = self.f(self.n) / (self.f(self.n - k) * (self.f(k)))
        second_factor = (self.p ** k) * ((1 - self.p) ** (self.n - k))
        return first_factor * second_factor

    def cdf(self, k):
        """returns cdf given k of successes"""
        if k < 0:
            return 0
        k = int(k)
        cdf = 0
        for k in range(k + 1):
            cdf += self.pmf(k)
        return cdf
