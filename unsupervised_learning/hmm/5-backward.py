#!/usr/bin/env python3
"""does backward alg for hidden markov model"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """does backward alg for hidden markov model"""

    T = len(Observation)
    N = Emission.shape[0]
    M = Emission.shape[1]

    B = np.zeros((N, T))

    B[:, T - 1] = 1

    for t in range(T - 2, -1, -1):
        for i in range(N):
            B[i, t] = np.sum(Transition[i, :] *
                             Emission[:, Observation[t + 1]]
                             * B[:, t + 1])

    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])

    return P, B
