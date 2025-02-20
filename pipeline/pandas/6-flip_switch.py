#!/usr/bin/env python3


def flip_switch(df):
    """sorts by reverse chronological order
    then transposes"""
    df_sorted = df.sort_values(by='Timestamp', ascending=False)
    df_transposed = df_sorted.transpose()
    return df_transposed