#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------------------------------------------------
# description
# ---------------------------------------------------------------

"""
2d_margin_perceptron.py
============
Please type "./2d_margin_perceptron.py -h" for usage help
    
Author:
    Li Hongjun

Description:
    This is a python3 script for the realization of classic perceptron
    and margin perceptron.

Reurirements:
    Python packages: numpy, matplotlib
"""

# ---------------------------------------------------------------
# import
# ---------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# function definition
# ---------------------------------------------------------------


def generate_random_sample(number, intercept, slope, margin,
                           x_min, x_max, y_min, y_max):
    """
    产生带标签的线性可分的随机样本，为实现线性可分，两类样本取自一条二维
    线上的随机点并加入纵向均匀分布的随机扰动，同时控制间隔实现线性可分    
    """
    class1_samples = []
    # 产生正类样本
    x1_list = [np.random.uniform(x_min, x_max, number)[i]
               for i in range(number)]
    y1_list = [x * slope + intercept + margin +
               np.random.uniform(y_min, y_max)
               for x in x1_list]
    class1_samples += ([x1_list[i], y1_list[i], 1]
                       for i in range(number))

    class2_samples = []
    # 产生负类样本
    x2_list = [np.random.uniform(x_min, x_max, number)[i]
               for i in range(number)]
    y2_list = [x * slope + intercept - margin +
               np.random.uniform(y_min, y_max)
               for x in x2_list]
    class2_samples += ([x2_list[i], y2_list[i], -1]
                       for i in range(number))
    return class1_samples, class2_samples


def classic_perceptron(data):
    arg_mat = [[_point[2], _point[2] * _point[0],
                _point[2] * _point[1]] for _point in data]
    # 构造规范化增广样本向量
    w = [0, 0, 0]
    # 初始化权向量
    k = 0

    while True:
        k += 1
        if k % 1000 == 0:
            print("Current iteration no: %d" % k)
        wrong_classfiy_c = 0
        for _point in arg_mat:
            if np.dot(w, _point) <= 0:
                w = list(np.add(w, _point))
                wrong_classfiy_c += 1
        if wrong_classfiy_c == 0:
            break
    return w


def margin_perceptron(data, gamma):
    arg_mat = [[_point[2], _point[2] * _point[0],
                _point[2] * _point[1]] for _point in data]
    # 构造规范化增广样本向量
    w = [0, 0, 0]
    # 初始化权向量
    k = 0

    while True:
        k += 1
        if k % 1000 == 0:
            print("Current iteration no: %d" % k)
        wrong_classfiy_c = 0
        for _point in arg_mat:
            if np.dot(w, _point) <= gamma:
                w = list(np.add(w, _point))
                wrong_classfiy_c += 1
        if wrong_classfiy_c == 0:
            break
    return w


def w_plot(w_optimized):
    """ 最优权向量方程 """
    return ([0, 1],
            [-w_optimized[0] / w_optimized[2],
             -(w_optimized[0] + w_optimized[1]) / w_optimized[2]])

# ---------------------------------------------------------------
# main function
# ---------------------------------------------------------------


def main():
    """ main """

    # 设置参数数生成两组线性可分的随机二维样本
    class1_samples, class2_samples = generate_random_sample(
        number=100, intercept=0.5, slope=3, margin=1,
        x_min=0, x_max=1.0,
        y_min=-0.5, y_max=0.5)

    # 绘制样本分布
    plt.figure(1, dpi=100)
    plt.scatter([class1_samples[i][0] for i in range(len(class1_samples))],
                [class1_samples[i][1] for i in range(len(class1_samples))],
                label="Class 1",
                c="b",
                s=10
                )
    plt.scatter([class2_samples[i][0] for i in range(len(class2_samples))],
                [class2_samples[i][1] for i in range(len(class2_samples))],
                label="Class -1",
                c="g",
                s=10
                )

    data = class1_samples + class2_samples
    # 测试数据集合

    #==========================Classic perceptron=============================
    w_optimized = classic_perceptron(data)
    plt.plot(w_plot(w_optimized)[0],
             w_plot(w_optimized)[1],
             label="Classic perceptron")
    #=========================================================================

    #==========================Gamma perceptron===============================
    for gamma in [0.1, 0.5, 1, 1.5, 4]:
        w_optimized = margin_perceptron(data, gamma)
        plt.plot(w_plot(w_optimized)[0],
                 w_plot(w_optimized)[1],
                 label="Gamma:%.2f" % gamma)
    #=========================================================================

    plt.title("Perceptron classifier")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
