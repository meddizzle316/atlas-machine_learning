#!/usr/bin/env python3
"""forward alg for hidden markov model"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """forward alg for hidden markov model"""
    T = len(Observation)
    N = Emission.shape[0]
    M = Emission.shape[0]

    F = np.zeros((N, T))
    F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    for t in range(1, T):
        for i in range(N):
            F[i, t] = np.sum(F[:, t - 1] * Transition[:, i]
                             * Emission[i, Observation[t]])

    P = np.sum(F[:, T - 1])
    return P, F
