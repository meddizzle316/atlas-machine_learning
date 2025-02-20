#!/usr/bin/env python3
"""drops nan columns"""


def prune(df):
    return df.dropna()