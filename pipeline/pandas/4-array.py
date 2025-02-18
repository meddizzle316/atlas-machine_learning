#!/usr/bin/env python3
"""converts df to np array"""


def array(df):
    """converts to numpy"""
    df = df[["High", "Close"]].tail(10)
    df = df.to_numpy()
    return df
