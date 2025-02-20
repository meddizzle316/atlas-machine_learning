#!/usr/bin/env python3
"""sorts df frame by high column"""


def high(df):
    """sorts by high"""
    return df.sort_values(by="High", ascending=False)
