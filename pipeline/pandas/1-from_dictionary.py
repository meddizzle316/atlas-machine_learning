#!/usr/bin/env python3
"""creates df from dictionary"""
import pandas as pd

row_names = ["A", "B", "C", "D"]
first_column = [0.0, 0.5, 1.0, 1.5]
second_column = ["one", "two", "three", "four"]

dict = {}
dict['First'] = first_column
dict['Second'] = second_column

df = pd.DataFrame(dict, index=row_names)
