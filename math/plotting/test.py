#!/usr/bin/env python3
"""learning how to use matplotlib"""
import matplotlib.pyplot as plt


fig = plt.figure(figsize=(10,8), dpi=100)

ax = fig.add_subplot(111)

ax.plot([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])

plt.show()
