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
