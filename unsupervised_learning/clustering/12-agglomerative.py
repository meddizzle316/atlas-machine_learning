#!/usr/bin/env python3
"""does agglometive clustering"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """does agglomerative clustering"""

    z = scipy.cluster.hierarchy.linkage(X, method='ward')
    dend = scipy.cluster.hierarchy.dendrogram(z)
    clss = scipy.cluster.hierarchy.fcluster(z, t=dist, criterion='distance')

    return clss
