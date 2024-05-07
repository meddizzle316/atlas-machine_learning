#!/usr/bin/env python3
"""plots a line"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """plots a line"""
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(y, color='r')
    plt.xlim(0, 10)
    plt.show()
