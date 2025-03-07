#!/usr/bin/env python3
"""loads pre made gymnasium environment    """
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False, render_mode="ansi"):
    """loads pre made gymnasium environment"""
    env = gym.make(
        "FrozenLake-v1",
        is_slippery=is_slippery,
        map_name=map_name,
        desc=desc,
        render_mode=render_mode)
    if map_name is None and is_slippery is None:
        env = gym.make("FrozenLake-v1", desc=desc, render_mode=render_mode)
    return env
