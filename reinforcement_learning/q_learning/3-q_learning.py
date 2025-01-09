#!/usr/bin/env python3
"""performs training loop for q learing"""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """training loop"""
    rewards_per_episode = np.zeros(episodes) # not sure what this is
    learning_rate_a = alpha # is this true?
    discount_factor_g = gamma # hopefully this translates
    for episode in range(episodes):
        state = env.reset()[0]
        terminated = False # Do I need this?
        truncated = False  # Do I need these?
        steps = 0
        while(not terminated and not truncated):
            action = epsilon_greedy(Q, state, epsilon)

            new_state, reward, terminated, truncated, _ = env.step(action)

            Q[state, action] = Q[state,action] + learning_rate_a * (
             reward + discount_factor_g * np.max(Q[new_state, :]) - Q[state, action]
            )

            state = new_state
            steps += 1 # I think this is what max steps means? The max steps for each 'episode' right?
            if steps >= max_steps:
                break

        epsilon = max(epsilon - epsilon_decay, 0)

        if  epsilon==0:
            learning_rate_a = min_epsilon

        if reward == 1:
            rewards_per_episode[episode] = 1

    env.close()

    # total_rewards = np.zeros(episodes)
    # for t in range(episodes):
    #     total_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])

    return Q, rewards_per_episode