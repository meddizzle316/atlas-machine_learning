#!/usr/bin/env python3
"""uses monte carlo method"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs Monte Carlo value estimation using incremental (every-visit) updates.

    Parameters
    ----------
    env : gymnasium.Env
        The environment instance.
    V : np.ndarray of shape (s,)
        The initial value estimates for each state s.
    policy : callable
        A function that takes in a state (int) and returns the next action (int) to take.
    episodes : int, optional (default=5000)
        Total number of episodes to train over.
    max_steps : int, optional (default=100)
        Maximum number of steps per episode.
    alpha : float, optional (default=0.1)
        The learning rate.
    gamma : float, optional (default=0.99)
        The discount factor.

    Returns
    -------
    V : np.ndarray of shape (s,)
        The updated value function after all episodes.
    """
    returns = {state: [] for state in range(env.observation_space.n)}
    for _ in range(episodes):
        # Reset the environment to start a new episode
        state, info = env.reset()

        # Track states and rewards for this episode
        episode = []

        # Generate an episode following the given policy
        while True:
            action = policy(state)
            next_state, reward, done, truncated, info = env.step(action)

            episode.append((state, reward))


            if done or truncated:
                break
            state = next_state

        # At the end of the episode, calculate the return and update V for each state visited
        G = 0.0
        # Traverse backwards to calculate returns
        for state, reward in reversed(episode):
            G = gamma * G + reward
            returns[state].append(G)
            V[state] = np.mean(returns[state])


    return V

