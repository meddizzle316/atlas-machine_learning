#!/usr/bin/env python3
"""creates df from csv"""
import pandas as pd


def from_file(filename, delimiter):
    """df from csv"""
    return pd.read_csv(filename, delimiter=delimiter)
