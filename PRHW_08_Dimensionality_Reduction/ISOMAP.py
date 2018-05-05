#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ---------------------------------------------------------------
# description
# ---------------------------------------------------------------

"""
ISOMAP.py
============
Please type "./ISOMAP.py -h" for usage help
    
Author:
    Li Hongjun

Description:
    This is a python3-realized ISOMAP dim reduction algorithm
    on a 3d-N-shape manifold.

Reurirements:
    Python packages: numpy, matplotlib
"""

# ---------------------------------------------------------------
# import
# ---------------------------------------------------------------

import argparse
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from queue import PriorityQueue

# ---------------------------------------------------------------
# function definition
# ---------------------------------------------------------------


def parse_args():
    """ master argument parser """
    parser = argparse.ArgumentParser(
        description="This is a python3-realized ISOMAP dim \
         reduction algorithm on a 3d-N-shape manifold.",
        # epilog="",
        # formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '-k', '--K_NN',
        type=int,
        required=False,
        default=20,
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
    """ 产生三维空间的N-形流形数据. """
    n = 500
    x_axis = np.random.rand(n)
    y_axis = 1.5 * (2 * np.random.rand(n) - 1) * np.pi + np.pi
    z_axis = np.sin(y_axis)
    x_axis.shape = y_axis.shape = z_axis.shape = (1, n)

    mf_samples = np.column_stack((x_axis.transpose(), y_axis.transpose(),
                                  z_axis.transpose()))
    mf_samples = mf_samples[mf_samples[:, 1].argsort()]
    return mf_samples


def cal_graph_g(data, k):
    """ Step 1: 计算近邻点(KNN)，
     计算每点路径长度得图G """
    n = data.shape[0]
    graph_g = np.zeros((n, n))
    _bool = graph_g == 0
    graph_g[_bool] = np.inf

    # 计算路径
    for i in range(n):
        _len = data.shape[0]
        dist = data - np.tile(data[i], (_len, 1))
        dist = np.linalg.norm(dist, axis=1)
        order = np.argsort(dist)
        dist = np.sort(dist)

        # 生成G，元素为距离
        dist, order = dist[1:k + 1], order[1:k + 1]
        graph_g[i, order] = dist
        graph_g[order, i] = dist
        graph_g[i, i] = 0
    return graph_g


def cal_shortest_path(graph_g, row):
    """ Step 2: 计算最短路径，对第row行（点）"""
    dist_row = np.copy(graph_g[row])
    final = [0 for i in range(dist_row.shape[0])]
    final[row] = 1

    _queue = PriorityQueue()
    for i in range(dist_row.shape[0]):
        _queue.put((dist_row[i], i))
    dist_row[row] = 0
    for i in range(dist_row.shape[0]):
        if sum(final) == dist_row.shape[0]:
            break
        if i != row:
            _iter = _queue.get()
            key = _iter[1]
            while final[key] != 0:
                _iter = _queue.get()
                key = _iter[1]
            k = key
            mini = dist_row[key]
            final[k] = 1
            for j in range(dist_row.shape[0]):
                if final[j] == 0 and (mini + graph_g[k, j] < dist_row[j]):
                    dist_row[j] = mini + graph_g[k, j]
                    _queue.put((dist_row[j], j))
    return dist_row


def mds(graph_g, p):
    """ Step 3: MDS计算低维映射，至p维空间 """
    graph_g = np.asarray(graph_g)
    dist_mat = graph_g.copy()
    for i in range(graph_g.shape[0]):
        dist = cal_shortest_path(graph_g, i)
        dist_mat[i, :] = dist
    d_square = dist_mat ** 2

    total_mean = np.mean(d_square)
    column_mean = np.mean(d_square, axis=0)
    row_mean = np.mean(d_square, axis=1)

    b_mat = np.zeros(d_square.shape)
    for i in range(b_mat.shape[0]):
        for j in range(b_mat.shape[1]):
            b_mat[i][j] = -0.5 * (d_square[i][j] - row_mean[i] - column_mean[j]
                                  + total_mean)

    eig_val, eig_vec = np.linalg.eig(b_mat)
    x = np.dot(eig_vec[:, :p], np.sqrt(np.diag(eig_val[:p])))
    return x


def isomap(data, k, p):
    """ ISOMAP """
    graph_g = cal_graph_g(data, k)
    n = data.shape[0]
    graph_g_cop = graph_g.copy()
    for i in range(n):
        dist_row = cal_shortest_path(graph_g, i)
        graph_g_cop[i] = dist_row
    return mds(graph_g_cop, p)


def make_plot(data_ori, data_red):
    """ 结果展示 """
    fig = plt.figure(1, dpi=150)
    ori_3d = fig.add_subplot(121, projection='3d')
    ori_3d.scatter(data_ori[:, 0], data_ori[:, 1], data_ori[:, 2], c=data_ori[:, 1])
    ori_3d.set_title('Origin data', fontsize=8)

    dim_red = fig.add_subplot(122)
    dim_red.scatter(data_red[:, 0], data_red[:, 1], c=data_ori[:, 1])
    dim_red.set_title('Dim reduced Data', fontsize=8)
    plt.suptitle('ISOMAP', fontsize=10)
    plt.show()


# ---------------------------------------------------------------
# main function
# ---------------------------------------------------------------


def main():
    """ main """
    args = parse_args()
    k = args.K_NN       # 近邻参数
    p = args.P_Dim      # 目标维数

    mf_samples = data_gen()
    data_red = isomap(mf_samples, k, p)

    make_plot(mf_samples, data_red)


if __name__ == "__main__":
    main()
