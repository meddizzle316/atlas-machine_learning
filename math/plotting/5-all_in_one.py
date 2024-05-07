#!/usr/bin/env python3
"""does 5 graphs in one"""
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def all_in_one():
    """plots five graphs in one"""
    y0 = np.arange(0, 11) ** 3

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    # fig, axs = plt.subplots(2, 2, figsize=(7,10))
    fig = plt.figure(figsize=(8, 5))
    gs = GridSpec(3, 2, figure=fig, wspace=0.3, hspace=0.9)
    fig.suptitle("All in One", fontsize='x-large')
    
    #top left graph
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(y0, color='r')
    ax1.set_xlim(0, 10)
    ax1.set_yticks(np.linspace(0, 1000, 3))
    # # top right graph; Scatter; Men's Height vs Weight

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(x1, y1, color='m')
    ax2.set_title("Men's Height vs Weight", fontsize='x-small')
    ax2.set_ylabel("Weight (lbs)", fontsize='x-small')
    ax2.set_xlabel("Height (in)", fontsize='x-small')
    ax2.set_xticks(np.linspace(60, 80, 3))
    ax2.set_yticks(np.linspace(170, 190, 3))
    # # middle left graph; just C-14

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.semilogy(x2, y2)
    ax3.set_ylabel("Fraction Remaining", fontsize='x-small')
    ax3.set_xlabel("Time (years)", fontsize='x-small')
    ax3.set_xlim(0, 28000)
    ax3.set_title("Exponential Decay of C-14", fontsize='x-small')
    ax3.set_xticks(np.linspace(0, 20000, 3))
    # # middle right graph; C-14 and Ra-226

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(x3, y31, color='red', linestyle='dashed', label='C-14')
    ax4.plot(x3, y32, color='green', label='Ra-226')
    ax4.set_ylabel("Fraction Remaining", fontsize='x-small')
    ax4.set_xlabel("Time (years)", fontsize='x-small')
    ax4.set_xlim(0, 20000)
    ax4.set_title("Exponential Decay of Radioactive Elements", fontsize='x-small')
    ax4.legend()


    ax5 = fig.add_subplot(gs[2, :])
    bins = np.linspace(0, 100, 11)
    ax5.hist(student_grades, bins=bins, edgecolor='black')
    ax5.set_xticks(bins)
    ax5.set_xlim(0, 100)
    ax5.set_title("Project A", fontsize='x-small')
    ax5.set_xlabel("Grades", fontsize='x-small')
    ax5.set_ylabel("Number of Students", fontsize='x-small')
    ax5.set_yticks(np.linspace(0, 30, 4))
    # plt.tight_layout()
    plt.show()
