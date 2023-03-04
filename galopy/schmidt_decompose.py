import numpy as np
import torch.linalg
import math


def tr_rho_sqrt(rho):
    return torch.diagonal(torch.matmul(rho, rho)).sum()


def partial_trace(matrix):
    return matrix.reshape(2, 2, 2, 2).transpose(0, 3).transpose(2, 3) \
        .diagonal(dim1=0, dim2=1).transpose(0, 2).transpose(2, 1).sum(0)


def rho_entropy(state):
    rho = torch.matmul(state.reshape(-1, 1).conj(), state.reshape(1, -1))
    return tr_rho_sqrt(partial_trace(rho))


# C = [[1., 0., 0., 0.],
#      [0., 1., 0., 0.],
#      [0., 0., 0., 1.],
#      [0., 0., 1., 0.]]
# C1 = torch.tensor([[0.0625, 0.125, 0.1875, 0.25],
#                    [0.125, 0.1875, 0.25, 0.3125],
#                    [0.1875, 0.25, 0.3125, 0.375],
#                    [0.25, 0.3125, 0.375, 0.4375]])
# C1 = np.array([[0.5, 0., 0., 0.5],
#                [0., 0., 0., 0.],
#                [0., 0., 0., 0.],
#                [0.5, 0., 0., 0.5]])
# r = 1 / math.sqrt(2)
# t = 1 / math.sqrt(3)
# C1 = torch.tensor([[r, 0, 0, r],
#                    [0, 0, 0, 0],
#                    [0, 0, 0, 0],
#                    [r, 0, 0, r]])
# A = torch.tensor([[1, 2, 3, 4],
#                   [5, 6, 7, 8],
#                   [9, 10, 11, 12],
#                   [13, 14, 15, 16]])
# array = torch.tensor([[[[0.3768 + 0.2980j, -0.0736 - 0.0051j, 0.0000 + 0.0000j, 0.0000 + 0.0000j],
#                         [0.0523 + 0.0562j, 0.4876 + 0.1098j, 0.0000 + 0.0000j, 0.0000 + 0.0000j],
#                         [0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.4887 + 0.1107j, -0.0716 + 0.0283j],
#                         [0.0000 + 0.0000j, 0.0000 + 0.0000j, -0.1410 - 0.0554j, -0.9611 + 0.2199j]]],
#
#                       [[[0.2914 + 0.4954j, 0.0000 + 0.0000j, 0.5402 + 0.5190j, 0.0000 + 0.0000j],
#                         [0.0000 + 0.0000j, -0.0279 - 0.1058j, 0.0000 + 0.0000j, 0.0000 + 0.0000j],
#                         [0.0742 - 0.2023j, 0.0000 + 0.0000j, 0.0032 - 0.0408j, 0.0000 + 0.0000j],
#                         [0.0000 + 0.0000j, 0.2133 - 0.2956j, 0.0000 + 0.0000j, -0.1072 + 0.2922j]]],
#
#                       [[[-0.4167 + 0.1075j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j],
#                         [0.0000 + 0.0000j, -0.9849 + 0.1421j, 0.0000 + 0.0000j, 0.0000 + 0.0000j],
#                         [0.0000 + 0.0000j, 0.0000 + 0.0000j, -0.6239 + 0.0498j, 0.0000 + 0.0000j],
#                         [0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.4323 + 0.0128j]]]])
# 1 2
# reshaped = A.reshape(2, 2, 2, 2).transpose(0, 3).transpose(2, 3)
# print(partial_trace(C1))
# print(tr_rho_sqrt(partial_trace(C1)))
# state = torch.tensor([r, 0., 0.5, 0.5])
# state = torch.tensor([0.7753 + 0.6131j, -0.1514 - 0.0105j, 0.0000 + 0.0000j, 0.0000 + 0.0000j])
# print("Entropy: ", rho_entropy(state))
# print(state.abs().square().sum())
# print(torch.matmul(state.reshape(-1, 1), state.reshape(1, -1)))

array = torch.tensor([[[-3.2950e-07 - 2.8480e-07j, 0.0000e+00 + 0.0000e+00j,
                        -2.1815e-04 + 5.0588e-04j, 0.0000e+00 + 0.0000e+00j],
                       [-3.9305e-01 + 3.0952e-01j, -1.1895e-01 + 4.8575e-01j,
                        4.1311e-01 + 2.8148e-01j, 4.9864e-01 - 3.2696e-02j],
                       [-1.9781e-01 + 4.5889e-01j, -1.2687e-01 - 4.8353e-01j,
                        -4.9754e-01 - 5.0600e-02j, 4.2328e-01 - 2.6669e-01j],
                       [0.0000e+00 + 0.0000e+00j, 0.0000e+00 + 0.0000e+00j,
                        1.5876e-04 - 5.2754e-04j, 7.3414e-08 + 1.8853e-07j]]])

new_array = array.reshape(4, 4).transpose(0, 1)
sums = new_array.abs().square().sum(1)
print(new_array)
print(sums)
# out = new_array / sums.unsqueeze(-1)
# print(out)
# output = [([(rho_entropy(vector)) for vector in matrix]) for matrix in out]
# print(output)

# (values1, ind1) = new_array.max(1)
# (values2, ind2) = new_array.min(1)
# print(values1.sub(values2))
# print(torch.tensor(0.7753 + 0.6131j, ).abs().square().sum())
