#!/usr/bin/env python3
"""drops nan columns"""


def prune(df):
    """drops nan columns"""
    return df.dropna()
