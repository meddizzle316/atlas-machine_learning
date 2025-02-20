#!/usr/bin/env python3
"""concats a df"""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """concats 2 dfs"""
    df1.set_index("Timestamp",inplace=True)
    df2.set_index("Timestamp", inplace=True)

    target_index = df2.index.get_loc(1417411920)
    df2_selected = df2[:target_index+1]

    df_combine = pd.concat([df2_selected, df1], axis=0, keys=["bitstamp", "coinbase"]) # might be 1?

    return df_combine