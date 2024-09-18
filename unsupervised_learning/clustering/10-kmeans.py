#!/usr/bin/env python3
"""uses modules to run kmeans"""
import numpy as np
import sklearn.cluster


def kmeans(X, k):
    """performs k-means on dataset"""
    if not isinstance(X, np.ndarray) or not isinstance(k, int):
        return None, None
    kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    return kmeans.cluster_centers_, kmeans.labels_
