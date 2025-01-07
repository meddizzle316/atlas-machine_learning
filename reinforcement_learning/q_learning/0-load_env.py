#!/usr/bin/env python3
"""loads pre made gymnasium environment    """
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """loads pre made gymnasium environment"""
    env = gym.make("FrozenLake-v1", is_slippery=is_slippery, map_name=map_name)
    if map_name is None and is_slippery is None:
        env = gym.make("FrozenLake-v1", map_name='8x8')
    return env