#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ---------------------------------------------------------------
# description
# ---------------------------------------------------------------

"""
LLE.py
============
Please type "./LLE.py -h" for usage help
    
Author:
    Li Hongjun

Description:
    This is a python3-realized LLE dim reduction algorithm
    on a 3d-3-shape manifold.

Reurirements:
    Python packages: numpy, matplotlib, sklearn
"""

# ---------------------------------------------------------------
# import
# ---------------------------------------------------------------

import argparse
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics


# ---------------------------------------------------------------
# function definition
# ---------------------------------------------------------------


def parse_args():
    """ master argument parser """
    parser = argparse.ArgumentParser(
        description="This is a python3-realized ISOMAP dim \
         reduction algorithm on a 3d-3-shape manifold.",
        # epilog="",
        # formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '-k', '--K_NN',
        type=int,
        required=False,
        default=13,
        help="""
        Define the para in 1-step KNN.
        """
    )
    parser.add_argument(
        '-p', '--P_Dim',
        type=int,
        required=False,
        default=2,
        help="""
        Define the objective dimension.
        Must less than the origin data dimension.
        """
    )

    args = parser.parse_args()
    return args


def data_gen():
    """ 产生三维空间的3-形流形数据. """
    n = 1000
    x_axis = np.random.rand(n) * 100
    z_axis = np.random.rand(n) * 100

    y_axis = np.multiply((-np.square(z_axis - 30) + 1000), (z_axis < 40) + 0) \
             + np.multiply((np.square(z_axis - 50) + 800),
                           np.logical_and(z_axis < 60, z_axis >= 40) + 0) \
             + np.multiply((-np.square(z_axis - 70) + 1000), (z_axis >= 60) + 0)
    x_axis.shape = y_axis.shape = z_axis.shape = (1, n)

    mf_samples = np.column_stack((x_axis.transpose(), y_axis.transpose(), z_axis.transpose()))
    mf_samples = mf_samples[mf_samples[:, 2].argsort()]
    return mf_samples


def cal_neighbors(data, k):
    """ Step 1: 计算每点k近邻(KNN) """

    samp_num = np.shape(data)[0]

    dist = metrics.pairwise.euclidean_distances(data)
    k_nearest = np.zeros((samp_num, samp_num))
    for i in range(samp_num):
        sorted_indexes = np.argsort(dist[i])
        k_nearest[i, sorted_indexes[1: k + 1]] = 1
    return k_nearest


def cal_weights(data, k_nearest, k):
    """ Step 2: 计算权重，满足约束下最小均方误差 """
    samp_num = np.shape(data)[0]
    weights = np.zeros((samp_num, samp_num))
    for i in range(samp_num):
        idxs = k_nearest[i].astype(bool)
        neighbors = data[idxs] - data[i]
        gram_mat = np.dot(neighbors, neighbors.transpose())
        w = np.dot(np.linalg.pinv(gram_mat), np.ones(k))
        w = w / np.sum(w)
        weights[i][idxs] = w
    return weights


def cal_embed_vec(weights, p):
    """ Step 3: 固定权重，计算低维映射，至p维空间 """

    num = np.shape(weights)[0]
    mat = np.dot((np.eye(num) - weights).transpose(), np.eye(num) - weights)
    (eigvals, eigvecs) = np.linalg.eigh(mat)
    y_bar = eigvecs[:, 1:p + 1]
    return y_bar


def lle(data, k, p):
    """ LLE """

    k_nearest = cal_neighbors(data, k)
    weights = cal_weights(data, k_nearest, k)
    return cal_embed_vec(weights, p)


def make_plot(data_ori, data_red):
    """ 结果展示 """
    fig = plt.figure(1, dpi=150)
    ori_3d = fig.add_subplot(121, projection='3d')
    ori_3d.scatter(data_ori[:, 0], data_ori[:, 1], data_ori[:, 2], c=data_ori[:, 2])
    ori_3d.set_title('Origin data', fontsize=8)

    dim_red = fig.add_subplot(122)
    dim_red.scatter(data_red[:, 0], data_red[:, 1], c=data_ori[:, 2])
    dim_red.set_title('Dim reduced Data', fontsize=8)
    plt.suptitle('LLE', fontsize=10)
    plt.show()


# ---------------------------------------------------------------
# main function
# ---------------------------------------------------------------


def main():
    """ main """
    args = parse_args()
    k = args.K_NN   # 近邻参数
    p = args.P_Dim  # 目标维数

    mf_samples = data_gen()
    data_red = lle(mf_samples, k, p)

    make_plot(mf_samples, data_red)


if __name__ == "__main__":
    main()
