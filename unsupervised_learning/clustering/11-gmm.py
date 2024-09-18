#!/usr/bin/env python3
"""getting GMM from dataset"""
import sklearn.mixture


def gmm(X, k):
    """getting GMM from dataset
    Returns: pi, m, S, clss, bic
    pi is a numpy.ndarray of shape (k,) containing
    the cluster priors
    m is a numpy.ndarray of shape (k, d) containing the
    centroid means
    S is a numpy.ndarray of shape (k, d, d) containing
    the covariance matrices
    clss is a numpy.ndarray of shape (n,) containing the
    cluster indices for each data point
    bic is a numpy.ndarray of shape (kmax - kmin + 1) containing
    the BIC value for each cluster size tested"""

    gm = sklearn.mixture.GaussianMixture(n_components=k).fit(X)
    pi = gm.weights_
    S = gm.covariances_
    m = gm.means_
    bic = gm.bic(X)
    clss = gm.predict(X)
    return pi, m, S, clss, bic
