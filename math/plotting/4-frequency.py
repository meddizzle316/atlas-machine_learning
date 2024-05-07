#!/usr/bin/env python3
"""plots grades"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """plots grades"""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # students = [0, 5, 10, 15, 20, 25, 30]
    bins = np.linspace(0, 100, 11)
    students = np.linspace(0, 30, 7)

    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")
    plt.hist(student_grades, bins=bins, edgecolor='black')
    plt.xticks(bins)
    plt.yticks(students)
    plt.xlim(0, 100)
    plt.show()
