#!/usr/bin/env python3
"""uses monte carlo method"""
import numpy as np


def monte_carlo(env, policy, episodes=5000, max_steps=100, gamma=0.99):
    """
    Monte Carlo evaluation with sample-average updates (First-Visit MC).

    Parameters:
    -----------
    env : gymnasium.Env
        The environment instance.
    policy : callable
        Function mapping state -> action.
    episodes : int
        Number of episodes to run.
    max_steps : int
        Max steps per episode.
    gamma : float
        Discount factor.

    Returns:
    --------
    V : np.ndarray
        Estimated value function (shape depends on env.observation_space.n).
    """
    # Assume discrete observation space
    num_states = env.observation_space.n

    # Initialize all values to 0
    V = np.zeros(num_states)

    # We’ll store all returns for each state to compute the mean
    returns_per_state = [[] for _ in range(num_states)]

    for _ in range(episodes):
        # Generate one episode
        episode = []
        state, _ = env.reset()

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, truncated, info = env.step(action)

            episode.append((state, reward))

            state = next_state
            if done or truncated:
                break

        # Now compute returns for each state visited in this episode
        visited_states = set()
        G = 0.0
        for t in reversed(range(len(episode))):
            s_t, r_t = episode[t]
            G = gamma * G + r_t

            # First-visit: only update if it's the *first time* we've seen s_t going backward
            if s_t not in visited_states:
                visited_states.add(s_t)
                returns_per_state[s_t].append(G)
                # Value estimate is average of all returns for s_t
                V[s_t] = np.mean(returns_per_state[s_t])

    return V

