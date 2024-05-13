#!/usr/bin/env python3
"""Attempting a Poisson Distribution"""


class Poisson:
    """Poisson Class"""
    def __init__(self, data=None, lambtha=1.):
        self.data = data
        if data != None:
            total_events = sum(data)
            if lambtha == 1.:
                lambtha_value = total_events / len(data)
                self.lambtha = lambtha_value
        if lambtha != 1.:
            self.lambtha = lambtha
        
        
