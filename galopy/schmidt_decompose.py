import numpy as np
import torch.linalg
import math
import random
# from population import RandomPopulation, FromFilePopulation


def tr_rho_sqrt(rho):
    return torch.diagonal(torch.matmul(rho, rho)).sum()


def partial_trace(matrix):
    return matrix.reshape(2, 2, 2, 2).transpose(0, 3).transpose(2, 3) \
        .diagonal(dim1=0, dim2=1).transpose(0, 2).transpose(2, 1).sum(0)


def rho_entropy(state):
    rho = torch.matmul(state.reshape(-1, 1).conj(), state.reshape(1, -1))
    return tr_rho_sqrt(partial_trace(rho))


CONST_ENTANGLEMENT_THRESHOLD = 0.2


def maximum_entanglement(matrix):
    res_vector = torch.tensor([0, 0])
    for vector in matrix:
        # print("what: ", vector)
        if res_vector[0] < CONST_ENTANGLEMENT_THRESHOLD and vector[0] > res_vector[0]:
            res_vector = vector
        if vector[0] > CONST_ENTANGLEMENT_THRESHOLD and vector[1] > res_vector[1]:
            res_vector = vector
    # print(res_vector)
    return res_vector.tolist()


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
r_ = 1 / math.sqrt(2)
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
state = torch.tensor([0.0625, 0.125, 0.1875, 0.25])
# state = state/(state.abs().square().sum().sqrt())
# print(state.abs().square().sum().sqrt())
# print(state)
# a = torch.tensor([[-0., -1.9353, -0.4605, -0.2917],
#                    [ 0.1815, -1.0111,  0.9805, -1.5923],
#                    [ 0.1062,  1.4581,  0.7759, -1.2344],
#                    [-0.1830, -0.0313,  1.1908, -1.4757]])
# b = torch.tensor([ 0.,  0.2930, -0.8113, -0.2308])
# print(torch.div(a, b))

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
# vector = torch.tensor([ 1./math.sqrt(2),0.,0.,1./math.sqrt(2)])
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
    [[0.5 + 0j, 0.5 + 0j, 0.5 + 0j, 0.5 + 0j], [0.5 + 0j, -0.5 + 0j, 0.5 + 0j, -0.5 + 0j],
     [0.5 + 0j, 0.5 + 0j, -0.5 + 0j, -0.5 + 0j], [0.5 + 0j, -0.5 + 0j, -0.5 + 0j, 0.5 + 0j]])
XZ = torch.inverse(ZX)

ZY = torch.tensor(
    [[0.5 + 0j, 0.5 + 0j, 0.5 + 0j, 0.5 + 0j], [0. + 0.5j, 0. - 0.5j, 0. + 0.5j, 0. - 0.5j],
     [0. + 0.5j, 0. + 0.5j, 0. - 0.5j, 0. - 0.5j], [-0.5 + 0j, 0.5 + 0j, 0.5 + 0j, -0.5 + 0j]])
YZ = torch.inverse(ZY)
# print(XY)
# print(XZ)
C1 = r_ * torch.tensor([[0. + 1j, 0. + 1j, 0., 0.],
                        [-1., 1., 0., 0.],
                        [0, 0., 1.j, 1.j],
                        [0., 0., -1., 1.]])
C2 = torch.tensor([[0.5 - 0.5j, 0.5 + 0.5j, 0., 0.],
                   [0., 0., 0., 0.],
                   [0, 0., 0, 0],
                   [0., 0., 0.5 - 0.5j, -0.5 - 0.5j]])
# vector2 = torch.
# print(torch.matmul(C2, ZX))
# print(torch.matmul(torch.matmul(XZ, C1), ZX))
# print(torch.matmul(torch.matmul(YZ, C1), ZY))
C_C = torch.tensor([[[0.0625, 0.125, 0.1875, 0.25],
                     [0.125, 0.1875, 0.25, 0.3125],
                     [0.1875, 0.25, 0.3125, 0.375],
                     [0.25, 0.3125, 0.375, 0.4375],
                     [0.1875, 0.25, 0.3125, 0.375],
                     [0.25, 0.3125, 0.375, 0.4375]
                     ],
                    [[0.0625, 0.125, 0.1875, 0.25],
                     [0.125, 0.1875, 0.25, 0.3125],
                     [0.1875, 0.25, 0.3125, 0.375],
                     [0.25, 0.3125, 0.375, 0.4375],
                     [0.1875, 0.25, 0.3125, 0.375],
                     [0.25, 0.3125, 0.375, 0.4375]
                     ],
                    [[0.0625, 0.125, 0.1875, 0.25],
                     [0.125, 0.1875, 0.25, 0.3125],
                     [0.1875, 0.25, 0.3125, 0.375],
                     [0.25, 0.3125, 0.375, 0.4375],
                     [0.1875, 0.25, 0.3125, 0.375],
                     [0.25, 0.3125, 0.375, 0.4375]
                     ],
                    [[0.0625, 0.125, 0.1875, 0.25],
                     [0.125, 0.1875, 0.25, 0.3125],
                     [0.1875, 0.25, 0.3125, 0.375],
                     [0.25, 0.3125, 0.375, 0.4375],
                     [0.1875, 0.25, 0.3125, 0.375],
                     [0.25, 0.3125, 0.375, 0.4375]
                     ]])
# some, _ = C_C.split([1, 3], dim=1)
# another = some.abs().square().sum(2).sqrt()
# print(C_C.split([4, 2], dim=1))
# print(torch.div(C_C, C_C))
# print(some)
# print(some/another.unsqueeze(-1))
# print(torch.matmul(XZ, C_C))
# print(torch.matmul(C_C, XZ))
# arr = np.array([[0.0349 - 0.0707j, 0.0000 + 0.0000j, -0.4772 - 0.2617j, 0.5187 + 0.3702j],
#                 [-0.1078 + 0.6118j, 0.1610 - 0.5347j, 0.0000 + 0.0000j, -0.0726 - 0.0253j],
#                 [0.0321 - 0.0699j, 0.0000 + 0.0000j, 0.4970 + 0.2543j, 0.5158 + 0.3462j],
#                 [0.0925 - 0.6305j, 0.1420 - 0.5255j, 0.0000 + 0.0000j, 0.0752 + 0.0238j]])
# arrt = arr.transpose().tolist()
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


# array = torch.tensor([arrt, arrt])


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
    print(normalized_states)
    entanglement_entropies = torch.tensor(
        [(1. - min(min([(rho_entropy(vector).abs().item()) for vector in matrix]), 1.)) for matrix in
         normalized_states])

    return basic_states_probabilities_match_result, entanglement_entropies, minimum


# (a, b, c) = some(array)
# print(a)
# print(b)
# print(c)
r_4_1 = 0.25 - 0.25j
r_4_2 = 0.25 + 0.25j
vector = torch.tensor([r_4_1, r_4_2, r_4_2, r_4_1])
# print(torch.matmul(ZX, vector))
# new_transforms_X = torch.tensor([[ 0.3929+0.1822j, -0.4314-0.0539j, -0.4369-0.0572j,  0.3880+0.1942j],
#         [-0.0881-0.2283j, -0.2224+0.0986j, -0.2171+0.1028j, -0.0833-0.2394j],
#         [-0.0276-0.2434j, -0.2434+0.0044j, -0.2399+0.0048j, -0.0207-0.2523j],
#         [ 0.2734-0.3432j,  0.3686-0.2378j,  0.3652-0.2390j,  0.2667-0.3351j]])
# sums_X = new_transforms_X.abs().square().sum(1).sqrt()
# out_X = new_transforms_X / vector
# print(out_X)
# print("res: ", vector.abs().square().sum(0).sqrt())
# print(rho_entropy(vector))
# print(0.7853* 0.7853)

array2 = torch.tensor([[1, 2, 3], [11, 22, 33], [111, 222, 333]])
array3 = torch.tensor([[1, 2, 3], [11, 22, 33], [111, 222, 333]])
array4 = torch.tensor([[[1., 1.],
                        [2., 2.]],

                       [[11., 11.],
                        [22., 22.]],

                       [[111., 111.],
                        [222., 222.]]])
array5 = torch.tensor([[[1., 1.],
                        [2., 2.]],

                       [[1., 11.],
                        [22., 22.]],

                       [[11., 111.],
                        [222., 22.]]])
# print(array4 - array5)
# print(C2.repeat(3, 1, 1).abs().sum(2).sum(1))
# print(torch.stack((array2, array3), dim=2))
# print(torch.tensor([1, 1, 1]) * 10)
# (a, b) = np.split(np.array([maximum_entanglement(matrix) for matrix in array4]), 2, axis=1)
# print(torch.tensor(a))
# print(torch.tensor(b))
# vector4 = torch.tensor([[1,2], [2, 3], [3, 4]])
# print(array4/vector4.unsqueeze(-1))
white_list = [0, math.acos(0), math.acos(1 / math.sqrt(2)), math.acos(1 / math.sqrt(3)), math.acos(1 / 2),
              math.acos(1 / math.sqrt(5)), math.asin(1 / math.sqrt(3)), math.asin(1 / math.sqrt(4)),
              math.asin(1 / math.sqrt(5))]
# print(57.295779513 * torch.tensor(white_list))
# print(random.choices(white_list, k=5))
# mask = torch.tensor([[[False, True],
#                       [False, True]],
#
#                      [[False, True],
#                       [False, True]],
#
#                      [[False, True],
#                       [True, True]]])
# array4[mask] = torch.tensor(random.choices(white_list, k=mask.sum().item())).reshape(
#             (mask.sum().item(),))
# print(array4)
# a = -0.1062-8.8662e-02j
# b = -0.0100-1.3184e-01j
# print(abs(a))
# print(abs(b))
# print((abs(a))*(abs(a)))
# print((abs(b))*(abs(b)))
# print((abs(a))*(abs(a)) + (abs(b))*(abs(b)))
# print(0.27241262793540955 * 0.27241262793540955)
# print(1/0.07420863985867587)
# print(2/0.07420863985867587)

# iswap = np.array([[1., 0., 0., 0.],
#                  [0., 0., 1.j, 0.],
#                  [0., 1.j, 0., 0.],
#                  [0., 0., 0., 1.]])
# evalues, evectors = np.linalg.eig(iswap)
# # Ensuring square root matrix exists
# print(evalues)
# print(evectors)
# assert (evalues >= 0).all()
# # sqrt_iswap = evectors * np.sqrt(evalues) @ np.linalg.inv(evectors)
# sqrt_iswap = torch.tensor([[1., 0., 0., 0.],
#                  [0., r_, r_*1.j, 0.],
#                  [0., r_*1.j, r_, 0.],
#                  [0., 0., 0., 1.]])
# print(sqrt_iswap)
#
# matrix = sqrt_iswap
# vector = torch.tensor([0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j]).unsqueeze(-1)
# result = torch.matmul(sqrt_iswap, vector)
# print(rho_entropy(result))
vector1 = torch.tensor([-0.216506, -0.125, 0, 0])
vector2 = torch.tensor([0.15625, -0.270632, 0, 0])
# print(vector1/(vector1.square().sum().sqrt()))
# print(vector2/(vector2.square().sum().sqrt()))
# vector3 = vector1/(vector1.square().sum().sqrt())
# vector4 = vector2/(vector2.square().sum().sqrt())
# vector5 = torch.tensor([-0.051472, 0.192097, 0.176472, 0.024409])
# print(vector5/(vector5.square().sum().sqrt()))
# vector6 = torch.tensor([-0.1928,  0.7195,  0.6609,  0.0914])
# print(rho_entropy(vector6))
# print(vector3.vdot(vector4).abs())
# some5 = FromFilePopulation(file_name="results/aux_ress_from_loqc/overnighter.txt")
# print(some5[0])
