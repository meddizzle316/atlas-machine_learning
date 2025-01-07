#!/usr/bin/env python3
"""does epsilon greedy operations"""""
import numpy as np

def epsilon_greedy(Q, state, epsilon):
    """doing epsilon greedy to balance exploration and exploitation """
    p = np.random.uniform()
    if p < epsilon:
        action = np.random.randint(4)
    else:
        action = np.argmax(Q[state,:]) # following the Q table

    return action
