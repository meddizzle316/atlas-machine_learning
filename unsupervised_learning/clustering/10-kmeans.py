#!/usr/bin/env python3
"""uses modules to run kmeans"""
import sklearn.cluster


def kmeans(X, k):
    """performs k-means on dataset"""
    kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    return kmeans.cluster_centers_, kmeans.labels_
