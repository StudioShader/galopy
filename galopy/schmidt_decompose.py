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

# array = torch.tensor([[[-3.2950e-07 - 2.8480e-07j, 0.0000e+00 + 0.0000e+00j,
#                         -2.1815e-04 + 5.0588e-04j, 0.0000e+00 + 0.0000e+00j],
#                        [-3.9305e-01 + 3.0952e-01j, -1.1895e-01 + 4.8575e-01j,
#                         4.1311e-01 + 2.8148e-01j, 4.9864e-01 - 3.2696e-02j],
#                        [-1.9781e-01 + 4.5889e-01j, -1.2687e-01 - 4.8353e-01j,
#                         -4.9754e-01 - 5.0600e-02j, 4.2328e-01 - 2.6669e-01j],
#                        [0.0000e+00 + 0.0000e+00j, 0.0000e+00 + 0.0000e+00j,
#                         1.5876e-04 - 5.2754e-04j, 7.3414e-08 + 1.8853e-07j]]])
#
# new_array = array.reshape(4, 4).transpose(1, 0)
# vector = torch.tensor([ 0.0000+0.0000j,  0.0000+0.0000j, -0.3654-0.4299j,  0.8169+0.1096j])
# print(rho_entropy(vector))
# print(rho_entropy(vector).item().real)
# print(rho_entropy(vector).abs().item())

# # res = torch.matmul()
# sums = new_array.abs().square().sum(1).sqrt()
# print(new_array)
# print(sums)
# # out = new_array / sums.unsqueeze(-1)
# out = new_array / sums
# print(out)
# output = [(rho_entropy(vector)) for vector in out]
# print(output)

# (values1, ind1) = new_array.max(1)
# (values2, ind2) = new_array.min(1)
# print(values1.sub(values2))
# print(torch.tensor(0.7753 + 0.6131j, ).abs().square().sum())


# Сделать преобразование с Z в X и обратно

# array = torch.tensor([[-3.2950e-07 - 2.8480e-07j, 0.0000e+00 + 0.0000e+00j,
#                         -2.1815e-04 + 5.0588e-04j, 0.0000e+00 + 0.0000e+00j],
#                        [-3.9305e-01 + 3.0952e-01j, -1.1895e-01 + 4.8575e-01j,
#                         4.1311e-01 + 2.8148e-01j, 4.9864e-01 - 3.2696e-02j],
#                        [-1.9781e-01 + 4.5889e-01j, -1.2687e-01 - 4.8353e-01j,
#                         -4.9754e-01 - 5.0600e-02j, 4.2328e-01 - 2.6669e-01j],
#                        [0.0000e+00 + 0.0000e+00j, 0.0000e+00 + 0.0000e+00j,
#                         1.5876e-04 - 5.2754e-04j, 7.3414e-08 + 1.8853e-07j]])
# vector = torch.tensor([0., 1., 0., 0.])
# print(r)
ZX = torch.tensor(
    [[0.5 + 0j, 0.5 + 0j, 0.5 + 0j, 0.5 + 0j], [0.5 + 0j, 0.5 + 0j, -0.5 + 0j, -0.5 + 0j],
     [0.5 + 0j, -0.5 + 0j, 0.5 + 0j, -0.5 + 0j], [0.5 + 0j, -0.5 + 0j, -0.5 + 0j, 0.5 + 0j]])
XZ = torch.inverse(ZX)

YX = torch.tensor(
    [[0.5 + 0j, 0.5 + 0j, 0.5 + 0j, 0.5 + 0j], [0. + 0.5j, 0. - 0.5j, 0. + 0.5j, 0. - 0.5j],
     [0. + 0.5j, 0. + 0.5j, 0. - 0.5j, 0. - 0.5j], [0. - 0.5j, 0. + 0.5j, 0. + 0.5j, 0. - 0.5j]])
XY = torch.inverse(YX)
# print(XY)
# print(XZ)
# print(torch.matmul(C1, XZ))
# print(torch.matmul(XZ, C1))
# C_C = torch.tensor([[[0.0625, 0.125, 0.1875, 0.25],
#                      [0.125, 0.1875, 0.25, 0.3125],
#                      [0.1875, 0.25, 0.3125, 0.375],
#                      [0.25, 0.3125, 0.375, 0.4375]],[[0.0625, 0.125, 0.1875, 0.25],
#                                                       [0.125, 0.1875, 0.25, 0.3125],
#                                                       [0.1875, 0.25, 0.3125, 0.375],
#                                                       [0.25, 0.3125, 0.375, 0.4375]], [[0.0625, 0.125, 0.1875, 0.25],
#                                                                                        [0.125, 0.1875, 0.25, 0.3125],
#                                                                                        [0.1875, 0.25, 0.3125, 0.375],
#                                                                                        [0.25, 0.3125, 0.375, 0.4375]],
#                     [[0.0625, 0.125, 0.1875, 0.25],
#                      [0.125, 0.1875, 0.25, 0.3125],
#                      [0.1875, 0.25, 0.3125, 0.375],
#                      [0.25, 0.3125, 0.375, 0.4375]]])
# print(torch.matmul(XZ, C_C))
# print(torch.matmul(C_C, XZ))
arr = np.array([[0.0349 - 0.0707j, 0.0000 + 0.0000j, -0.4772 - 0.2617j, 0.5187 + 0.3702j],
       [-0.1078 + 0.6118j, 0.1610 - 0.5347j, 0.0000 + 0.0000j, -0.0726 - 0.0253j],
       [0.0321 - 0.0699j, 0.0000 + 0.0000j, 0.4970 + 0.2543j, 0.5158 + 0.3462j],
       [0.0925 - 0.6305j, 0.1420 - 0.5255j, 0.0000 + 0.0000j, 0.0752 + 0.0238j]])
arrt = arr.transpose().tolist()
array = torch.tensor([[[-0.4194 - 0.2724j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, -0.4491 + 0.2195j],
                       [-4.9442e-01 - 7.4729e-02j, 3.3433e-09 + 1.9452e-08j,
                        0.0000e+00 + 0.0000e+00j, 3.1829e-01 - 3.8556e-01j],
                       [-2.5072e-01 - 4.3256e-01j, 0.0000e+00 + 0.0000e+00j,
                        -3.5052e-09 + 1.0070e-08j, 4.9998e-01 + 7.3966e-03j],
                       [-4.0700e-01 - 2.9022e-01j, -1.8673e-05 + 6.0529e-05j,
                        5.7600e-06 + 6.3081e-05j, -4.5842e-01 + 1.9993e-01j]],
                      [[-0.4194 - 0.2724j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, -0.4491 + 0.2195j],
                       [-4.9442e-01 - 7.4729e-02j, 3.3433e-09 + 1.9452e-08j,
                        0.0000e+00 + 0.0000e+00j, 3.1829e-01 - 3.8556e-01j],
                       [-2.5072e-01 - 4.3256e-01j, 0.0000e+00 + 0.0000e+00j,
                        -3.5052e-09 + 1.0070e-08j, 4.9998e-01 + 7.3966e-03j],
                       [-4.0700e-01 - 2.9022e-01j, -1.8673e-05 + 6.0529e-05j,
                        5.7600e-06 + 6.3081e-05j, -4.5842e-01 + 1.9993e-01j]]
                      ])
array = torch.tensor([arrt, arrt])


def some(transforms):
    reshaped = transforms.reshape(transforms.size()[0], 4, 4)
    new_transforms = reshaped.transpose(1, 2)
    print("new_transforms: ", new_transforms)
    print("abs sqr: ", new_transforms.abs().square())
    sums = new_transforms.abs().square().sum(2).sqrt()
    (values_max1, ind1) = sums.max(1)
    (values_min1, ind2) = sums.min(1)
    basic_states_probabilities_match_X = torch.ones(transforms.size()[0]).sub(values_max1.sub(values_min1))
    print("sums: ", sums)
    new_transforms_Z = torch.matmul(torch.matmul(ZX, reshaped), XZ).transpose(1, 2)
    print("new_transforms_Z: ", new_transforms_Z)
    sums_Z = new_transforms_Z.abs().square().sum(2).sqrt()
    (values_max2, ind1) = sums_Z.max(1)
    (values_min2, ind2) = sums_Z.min(1)
    basic_states_probabilities_match_Z = torch.ones(transforms.size()[0]).sub(values_max2.sub(values_min2))
    print("sums_Z: ", sums_Z)
    new_transforms_Y = torch.matmul(torch.matmul(YX, reshaped), XY).transpose(1, 2)
    print("new_transforms_Y: ", new_transforms_Y)
    sums_Y = new_transforms_Y.abs().square().sum(2).sqrt()
    (values_max3, ind1) = sums_Y.max(1)
    (values_min3, ind2) = sums_Y.min(1)
    basic_states_probabilities_match_Y = torch.ones(transforms.size()[0]).sub(values_max3.sub(values_min3))
    print("sums_Y: ", sums_Y)
    maximum = torch.maximum(values_max3, torch.maximum(values_max1, values_max2))
    minimum = torch.minimum(values_min3, torch.minimum(values_min1, values_min2))
    basic_states_probabilities_match_result = torch.ones(transforms.size()[0]).sub(maximum.sub(minimum))

    # calculate maximum entanglement of states
    # TODO: OPTIMIZE for torch
    normalized_states = new_transforms / sums.unsqueeze(-1)
    entanglement_entropies = torch.tensor(
        [(1. - min(min([(rho_entropy(vector).abs().item()) for vector in matrix]), 1.)) for matrix in
         normalized_states])

    return basic_states_probabilities_match_result, entanglement_entropies, minimum


# (a, b, c) = some(array)
# print(a)
# print(b)
# print(c)
# vector = torch.tensor([-0.4362 - 0.2701j, -1.2090 - 0.9063j, -1.2090 - 0.9063j, -0.4362 - 0.2702j])
# print("res: ", vector.abs().square().sum(0).sqrt())
# print(rho_entropy(vector))
# print(0.7853* 0.7853)
