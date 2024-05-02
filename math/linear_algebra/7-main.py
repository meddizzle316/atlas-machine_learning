#!/usr/bin/env python3

cat_matrices2D = __import__('7-gettin_cozy').cat_matrices2D

mat1 = [[1, 2], [3, 4]]
mat2 = [[5, 6]]
mat3 = [[7], [8]]
mat4 = cat_matrices2D(mat1, mat2)
mat5 = cat_matrices2D(mat1, mat3, axis=1)
# mat1[0] = [9, 10]
# mat1[1].append(5)
# mat6 = [[]]  corresponds to check 1
# mat7 = [[], []] corresponds to check 3
# mat8 = [] corresponds to check 2
mat9 = cat_matrices2D(mat8, mat1)
print(mat9)
print(mat1)
print(mat4)
print(mat5)
