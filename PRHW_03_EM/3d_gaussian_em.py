#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
3d_gaussian_em.py
============
Author:
    Li Hongjun

Description (zh-cn):·
    This is a python3 script of EM for estimation of the 3d-gaussian args.

Reurirements:
    Python packages: numpy
"""

import math
import numpy as np


def cal_mle(_x, _mu, _sigma):
    """ 计算单个样本的似然函数 """
    _x.shape = _mu.shape = (1, 3)
    _x = _x.transpose()
    _mu = _mu.transpose()
    # 转为列向量
    Lx = 1 / ((2 * math.pi) ** 1.5 * np.linalg.det(_sigma) ** 0.5) \
        * math.exp(-0.5 * (_x - _mu).transpose() @ np.linalg.inv(_sigma) @ (_x - _mu))
    return Lx


def e_step(incomplete_data, mu, sigma):
    """ 迭代E步，基于M步的参数值，求使似然函数最大的不完全样本的x_3，获得完整数据 """
    for i in range(5):
        x_3_temp = 0
        lx_prev = 0
        for j in np.arange(-10, 10, 0.01):
            j = round(j, 2)
            incomplete_data[i][2] = j
            lx = cal_mle(incomplete_data[i], mu, sigma)
            if lx >= lx_prev:
                x_3_temp = j
            lx_prev = lx
        incomplete_data[i][2] = x_3_temp
    return incomplete_data


def m_step(complete_data, incomplete_data):
    """ 迭代M步，基于E步获得的完整数据求参数mu和协方差阵sigma的期望 """
    whole_array = np.vstack((complete_data, incomplete_data))

    mu = np.mean(whole_array, axis=0)
    # 分别计算x_1,x_2,x_3的均值作为mu的期望

    sigma_sum = np.zeros((3, 3))
    for i in range(10):
        x = whole_array[i]
        for j in range(3):
            for k in range(3):
                sigma_sum[j][k] += (x[j] - mu[j]) * (x[k] - mu[k])

    # 协方差阵的期望
    sigma_mean = sigma_sum / 10
    return mu, sigma_mean


def mle_mu_sigma(complete_data):
    """ 由完整数据计算参数的极大似然估计 """
    whole_array = complete_data
    mu = np.mean(whole_array, axis=0)
    # 分别计算x_1,x_2,x_3的均值作为mu的期望

    sigma_sum = np.zeros((3, 3))
    for i in range(10):
        x = whole_array[i]
        for j in range(3):
            for k in range(3):
                sigma_sum[j][k] += (x[j] - mu[j]) * (x[k] - mu[k])

    # 协方差阵的期望
    sigma_mean = sigma_sum / 10
    return mu, sigma_mean


def main():
    """ main """
    samples = np.array([[0.42, -0.087, 0.58],
                        [-0.2, -3.3, -3.4],
                        [1.3, -0.32, 1.7],
                        [0.39, 0.71, 0.23],
                        [-1.6, -5.3, -0.15],
                        [-0.029, 0.89, -4.7],
                        [-0.23, 1.9, 2.2],
                        [0.27, -0.3, -0.87],
                        [-1.9, 0.76, -2.1],
                        [0.87, -1.0, -2.6]])

    # 参数赋初值
    mu = np.zeros((3))
    sigma = np.identity(3)

    incomplete_data = np.vstack((samples[2 * i + 1] for i in range(5)))
    incomplete_data[:, 2] = 0
    # 去掉x_3信息
    complete_data = np.vstack((samples[2 * i] for i in range(5)))
    # 利用参数初值计算缺失数据初值

    # 迭代参数
    eps = 1e-6
    _iter = 1
    mu_prev = np.zeros((3))
    sigma_prev = np.zeros((3, 3))

    # 开始迭代，以两轮迭代差值矩阵的二范数作为判断标准
    # while _iter <= 500:
    while _iter == 1 or np.linalg.norm(mu - mu_prev, ord=2) > eps or \
            np.linalg.norm(sigma - sigma_prev, ord=2) > eps:
        if _iter % 10 == 0:
            print("迭代次数: %d; mu改变量: %f; sigma改变量: %f;" %
                  (_iter, np.linalg.norm(mu - mu_prev, ord=2),
                   np.linalg.norm(sigma - sigma_prev, ord=2)))
        # E-step
        incom_2_com_data = e_step(incomplete_data, mu, sigma)
        mu_prev, sigma_prev = mu, sigma

        # M-step
        mu, sigma = m_step(complete_data, incom_2_com_data)
        _iter += 1

    # 迭代终止后
    print("迭代次数: %d; mu改变量: %f; sigma改变量: %f;\n" %
          (_iter, np.linalg.norm(mu - mu_prev, ord=2),
           np.linalg.norm(sigma - sigma_prev, ord=2)))
    print("mu EM估计结果:\n", mu, "\nsigma EM估计结果:\n", sigma)

    # 完整数据结果
    mu, sigma = mle_mu_sigma(samples)
    print("mu MLE结果:\n", mu, "\nsigma MLE结果:\n", sigma)
    # print(incom_2_com_data)


if __name__ == "__main__":
    main()


#=======指定收敛条件的结果===============
# 迭代次数: 3; x_l改变量: 0.000000; x_u改变量: 0.000000;
# x_l EM估计结果:
#  [-0.4    0.054 -0.18 ]
# x_u EM估计结果:
#  [0.38  0.69  0.089]
# x_l MLE结果:
#  [-0.4    0.054 -0.18 ]
# x_u MLE结果:
#  [0.38 0.69 0.12]
# [[-0.31    0.27   -0.0455]
#  [-0.15    0.53   -0.0455]
#  [ 0.17    0.69   -0.0455]
#  [-0.27    0.61   -0.0455]
#  [-0.12    0.054  -0.0455]]
#=======================================
