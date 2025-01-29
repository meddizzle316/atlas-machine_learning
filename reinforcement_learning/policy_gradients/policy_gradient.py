#!/usr/bin/env python3
"""computes simple policy"""
import numpy as np


def softmax(x):
    """performs softmax function as numpy doesn't have a built in one"""
    logits = np.exp(x - np.max(x))
    return logits / np.sum(logits, axis=1, keepdims=True)


def policy(matrix, weight):
    """computes policy with a weight of a matrix. You just matmul and then
    apply softmax"""
    base = np.matmul(matrix, weight)

    if base.ndim == 1:
        base = base[np.newaxis, :]

    policy_softmax = softmax(base)

    return policy_softmax


def policy_action(policy_softmax):
    """function that gets the policy of the given policy output"""
    action_probabilites = policy_softmax.squeeze(
        0)  # since output of policy is 2d and we need 1d

    num_possible_actions = len(action_probabilites)
    # choose an action
    action = np.random.choice(
        np.arange(num_possible_actions), p=action_probabilites
    )
    return action, action_probabilites


def policy_gradient(state, weight):
    """computes Monte Carlo policy gradient, returns the action
    and the log of the gradient"""
    policy_softmax = policy(
        state,
        weight)
    # applying softmax to make sure it's a valid probability
    # distribution (positive and sums to 1)

    action, action_probs = policy_action(policy_softmax)

    dlogpi = np.outer(state, np.eye(len(action_probs))[action] - action_probs)

    return action, dlogpi
