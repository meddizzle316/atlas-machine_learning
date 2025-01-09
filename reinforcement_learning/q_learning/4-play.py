#!/usr/bin/env python3
import numpy as np


def play(env, Q, max_steps=100):
    """plays using trained model for a given episode
    returns total rewards for the episode
    and list a rendered outputs representing the
    board state at each step"""

    # things the play (or evaluate) shouldn't do
    # 1. updated q table
    # 2. Do any exploration (only following q table)
    # 3. initialize the q table

    for i in range(max_steps):
        state = env.reset()[0]
        terminated = False
        truncated = False
        list_of_rendered_states = []

        while (not terminated and not truncated):
            action = np.argmax(Q[state,:])
            list_of_rendered_states.append(env.render())
            new_state, reward, terminated, truncated, _ = env.step(action)

            state = new_state
    # print(list_of_rendered_states)
    return reward, list_of_rendered_states