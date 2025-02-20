#!/usr/bin/env python3
"""slices given df, getting every 60th"""


def slice(df):
    """slices a df"""
    df = df[["High", "Low", "Close", "Volume_(BTC)"]]
    df = df.iloc[::60]
    return df
