#!/usr/bin/env python3
"""initializes q table"""
import numpy as np


def q_init(env):
    """initilizaes q table"""
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.n

    q_table = np.zeros((state_space_size, action_space_size))
    return q_table
