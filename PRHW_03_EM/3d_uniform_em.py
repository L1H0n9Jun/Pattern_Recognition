#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
3d_uniform_em.py
============
Author:
    Li Hongjun

Description (zh-cn):
    This is a python3 script of EM for estimation of the 3d-uniform args.
    
Reurirements:
    Python packages: numpy
"""

import numpy as np


def e_step(incomplete_data, _x_l, _x_u):
    """ 迭代E步，基于M步的参数值，求使似然函数最大的不完全样本的x_3，获得完整数据 """
    x_3_exp = (_x_l[2] + _x_u[2]) / 2
    incomplete_data[:, 2] = x_3_exp
    return incomplete_data


def m_step(complete_data, incomplete_data):
    """ 迭代M步，基于E步获得的完整数据求参数mu和协方差阵sigma的期望 """
    whole_array = np.vstack((complete_data, incomplete_data))
    _x_l = np.min(whole_array, axis=0)
    _x_u = np.max(whole_array, axis=0)
    # 分别计算max(x_i)作为x_u_i和min(x_i)作为x_l_i期望
    return _x_l, _x_u


def mle_xl_xu(complete_data):
    """ 由完整数据计算参数的极大似然估计 """
    whole_array = complete_data
    _x_l = np.min(whole_array, axis=0)
    _x_u = np.max(whole_array, axis=0)
    return _x_l, _x_u


def main():
    """ main """
    samples = [[-0.4, 0.58, 0.089],
               [-0.31, 0.27, -0.04],
               [0.38, 0.055, -0.035],
               [-0.15, 0.53, 0.011],
               [-0.35, 0.47, 0.034],
               [0.17, 0.69, 0.1],
               [-0.011, 0.55, -0.18],
               [-0.27, 0.61, 0.12],
               [-0.065, 0.49, 0.0012],
               [-0.12, 0.054, -0.063]]

    # 参数赋初值
    x_l = np.array([-2, -2, -2])
    x_u = np.array([2, 2, 2])

    incomplete_data = np.vstack((samples[2 * i + 1] for i in range(5)))
    incomplete_data[:, 2] = 0
    # 去掉x_3信息
    complete_data = np.vstack((samples[2 * i] for i in range(5)))
    # 利用参数初值计算缺失数据初值

    # 迭代参数
    eps = 1e-6
    _iter = 1
    x_l_prev = np.zeros((3))
    x_u_prev = np.zeros((3))

    # 开始迭代，以两轮迭代差值矩阵的二范数作为判断标准
    # while _iter <= 500:
    while np.linalg.norm(x_l - x_l_prev, ord=2) > eps and \
            np.linalg.norm(x_u - x_u_prev, ord=2) > eps:
        if _iter % 10 == 0:
            print("迭代次数: %d; x_l改变量: %f; x_u改变量: %f;" %
                  (_iter, np.linalg.norm(x_l - x_l_prev, ord=2),
                   np.linalg.norm(x_u - x_u_prev, ord=2)))

        # E-step
        incom_2_com_data = e_step(incomplete_data, x_l, x_u)
        x_l_prev, x_u_prev = x_l, x_u

        # M-step
        x_l, x_u = m_step(complete_data, incom_2_com_data)
        _iter += 1

    # 迭代终止后
    print("迭代次数: %d; x_l改变量: %f; x_u改变量: %f;" %
          (_iter, np.linalg.norm(x_l - x_l_prev, ord=2),
           np.linalg.norm(x_u - x_u_prev, ord=2)))
    print("x_l EM估计结果:\n", x_l, "\nx_u EM估计结果:\n", x_u)

    # 完整数据结果
    x_l, x_u = mle_xl_xu(samples)
    print("x_l MLE结果:\n", x_l, "\nx_u MLE结果:\n", x_u)
    print(incom_2_com_data)


if __name__ == "__main__":
    main()
