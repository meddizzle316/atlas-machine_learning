#!/usr/bin/env python3
"""renames a dataframe"""
import pandas as pd


def rename(df):
    """renames a df"""
    df = df.rename(columns={"Timestamp":"Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit='s')
    return df[['Datetime', 'Close']]
