#!/usr/bin/env python3
"""converts numpy array to dataframe"""
import string

import pandas as pd


def from_numpy(array):
    """converts pd df from numpy"""

    df = pd.DataFrame(array)
    alphabet = string.ascii_uppercase
    new_columns = {df.columns[i]: alphabet[i] for i in range(len(df.columns))}
    df = df.rename(columns=new_columns)

    return df