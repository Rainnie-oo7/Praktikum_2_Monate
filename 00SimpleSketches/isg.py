import torch
arr = torch.tensor([[118.], [661.], [290.], [716.]])
arr.reshape(1,4)
print(arr)
print(arr.shape)
# import numpy as np
# Ein Array mit 3 Zeilen und 3 Spalten
# arr = np.array([7, 8, 9])
# Zugriff auf die erste Zeile
# result = arr[-1]
# print(result)


# Ein 3D-Array mit Shape (2, 3, 3)
# arr_3d = np.array([
#     [
#         [1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9]
#     ],
#     [
#         [10, 11, 12],
#         [13, 14, 15],
#         [16, 17, 18]
#     ]
# ])
#
# # Zugriff auf die zweite Spalte (Index 1) in jeder "Ebene"
# result_3d = arr_3d[0, :, :]
# print(result_3d)
