#!/usr/bin/env python3
"""uses monte carlo method"""
import numpy as np
import random
import gymnasium as gym


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """runs monte carlo method"""

    stateNumber = env.observation_space.n

    sumReturnForEveryState = np.zeros(stateNumber)
    numberVisitsForEveryState = np.zeros(stateNumber)
    valueFunctionsEstimate = V

    for episode in range(episodes):
        visitedStatesInEpisode = []

        rewardInVisitedState = []
        (currentState, prob) = env.reset()
        visitedStatesInEpisode.append(currentState)

        print(f"simulating episode {episode}")

        while True:
            randomAction = env.action_space.sample()

            (currentState, currentReward, done, _, _) = env.step(randomAction)

            rewardInVisitedState.append(currentReward)

            if not done:
                visitedStatesInEpisode.append(currentState)

            else:
                break

        numberofVisitedStates = len(visitedStatesInEpisode)

        Gt = 0

        for current_episode in range(numberofVisitedStates - 1, -1, -1):
            stateTmp = visitedStatesInEpisode[current_episode]
            returnTmp = rewardInVisitedState[current_episode]

            Gt = gamma * Gt + returnTmp

            if stateTmp not in visitedStatesInEpisode[0:current_episode]:
                numberVisitsForEveryState[stateTmp] = numberVisitsForEveryState[stateTmp] + 1

                sumReturnForEveryState[stateTmp] = sumReturnForEveryState[stateTmp] + Gt

    for indexSum in range(stateNumber):
        if numberVisitsForEveryState[indexSum] != 0:
            valueFunctionsEstimate[indexSum] = sumReturnForEveryState[indexSum] / numberVisitsForEveryState[indexSum]

    return valueFunctionsEstimate
