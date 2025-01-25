#!/usr/bin/env python3
"""performs td algorithm"""
import numpy as np

def td_lambtha(
        env,
        V,
        policy,
        lambtha,
        num_episodes=5000,
        max_steps=100,
        alpha=0.1,
        gamma=0.99):
    """performs TD algorithm"""

    # apparently has same shape as V function, initialize to 0
    E = np.zeros_like(V)

    for session in range(num_episodes):
        # Generate one episode
        E.fill(0)  # resetting E
        state, _ = env.reset()

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, truncated, info = env.step(action)

            # Temporal Difference Error via rule
            t_d = reward + (gamma * (V[next_state]) - V[state])
            # I guess env.step(action) always gives r_t+1 ?

            E[state] += 1  # update Eligibility trace
            V += alpha * t_d * E  # update V using update rule
            # note that here we're updating all V
            # states, even if we didn't encounter them
            # in that step

            E *= gamma * lambtha  # Temporal Decay using gamma and lambtha

            state = next_state
            if done or truncated:
                break

    return V
