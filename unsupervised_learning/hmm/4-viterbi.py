#!/usr/bin/env python3
"""does viterbi alg"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """does viterbi alg"""


    T = len(Observation)
    N = Emission.shape[0]
    M = Emission.shape[1]

    delta = np.zeros((N, T))
    psi = np.zeros((N, T))

    delta[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    for t in range(1, T):
        for i in range(N):
            delta[i, t] = np.max(delta[:, t-1] * Transition[:, i] * Emission[i, Observation[t]])
            psi[i, t] = np.argmax(delta[:, t-1] * Transition[:, i])

    path = [np.argmax(delta[:, T-1])]

    for t in range(T-2, -1, -1):
        path.insert(0, int(psi[path[0], t+1]))
    P = np.max(delta[:, T-1])
    return path, P