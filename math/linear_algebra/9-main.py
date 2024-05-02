#!/usr/bin/env python3
import numpy as np

doo = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# Select the first and last (3rd row)
first_and_last_row = doo[[0, 2], :]
print(first_and_last_row)

# Select the first and last columns
first_and_last_columns = doo[:, [0, 2]]
print(first_and_last_columns)

# Combine the sliced arrays into a new 2D matrix
# combined_matrix = np.vstack((first_and_last_row, first_and_last_columns))

# print(combined_matrix)
