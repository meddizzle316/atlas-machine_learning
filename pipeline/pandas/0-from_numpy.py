#!/usr/bin/env python3
"""converts numpy array to dataframe"""

import pandas as pd


def from_numpy(array):
    """converts pd df from numpy"""

    df = pd.DataFrame(array)
    alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I",
                "J", "K", "L", "M", "N", "O", "P", "Q", "R",
                "S", "T", "U", "V", "W", "X", "Y", "Z"]
    new_columns = {df.columns[i]: alphabet[i] for i in range(len(df.columns))}
    df = df.rename(columns=new_columns)

    return df
