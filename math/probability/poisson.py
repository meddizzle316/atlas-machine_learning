#!/usr/bin/env python3
"""Attempting a Poisson Distribution"""


class Poisson:
    """Poisson Class"""
    def __init__(self, data=None, lambtha=1.):
        self.data = data
        if data != None:
            total_events = sum(data)
            lambtha_value = total_events / len(data)
            self.lambtha = float(lambtha_value)
        else:
            self.lambtha = float(lambtha)
        
        
