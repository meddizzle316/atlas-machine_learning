#!/usr/bin/env python3
"""makes new index"""


def index(df):
    df.set_index("Timestamp",inplace=True)
    return df