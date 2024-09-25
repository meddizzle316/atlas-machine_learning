#!/usr/bin/env python3
"""markov chain probabity"""
import numpy as np


def markov_chain(P, s, t=1):
    """markov chain probability"""

    for i in range(t):
        s = s @ P
    return s
