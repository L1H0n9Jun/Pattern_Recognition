#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ---------------------------------------------------------------
# description
# ---------------------------------------------------------------

"""
PRHW_02_Parameter_Estimation.parzen_window.py
============
Please type "./PRHW_02_Parameter_Estimation.parzen_window.py -h" for usage help
    
Author:
    Li Hongjun, 2017310864

Description:
    This is a python3 script for the realization of parzen window.

Reurirements:
    Python3 packages: numpy, matplotlib
"""

# ---------------------------------------------------------------
# import
# ---------------------------------------------------------------

import sys
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# function definition
# ---------------------------------------------------------------


def parse_args():
    """ master argument parser """
    parser = argparse.ArgumentParser(
        description="",
        # epilog="",
        # formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '-n', '--sample_number',
        type=str,
        required=True,
        help="""
        Number of samples used to estimate the population. 
        Could be a single int number or multiple numbers 
        separated by comma, on which the final figure depends.
        """
    )
    parser.add_argument(
        '-w', '--window_width',
        type=str,
        required=True,
        help="""
        Parzen window width used.
        Could be a single float number or multiple numbers 
        separated by comma, on which the final figure depends.
        """
    )
    parser.add_argument(
        '-t', '--window_type',
        type=str,
        required=True,
        choices=['uniform', 'gaussian', 'exponential'],
        help="""
        Parzen window type used.
        """
    )

    args = parser.parse_args()
    return args


def normal_random_gen(_mu_1, _sigma_1, _mu_2, _sigma_2, n):
    """ 从p(x)群体产生n个随机样本 """
    _rand_nor_1 = np.random.normal(_mu_1, _sigma_1, n)
    _rand_nor_2 = np.random.normal(_mu_2, _sigma_2, n)
    sample_list = []
    for i in range(0, n):
        sample_list.append(0.2 * _rand_nor_1[i]
                           + 0.8 * _rand_nor_2[i])
        # 此处随机数均取两位小数
    sample_list.sort()
    return sample_list


def unifrom_win_func(width_a, x):
    """ 定义方窗函数 """
    if abs(x / width_a) <= 0.5 * width_a:
        return 1.0 / width_a
    else:
        return 0


def gaussian_win_func(width_a, x):
    """ 定义正态窗函数 """
    pd = 1 / math.sqrt(2 * math.pi) * math.exp(
        -0.5 * (x / width_a) ** 2)
    return pd


def exponential_win_func(width_a, x):
    """ 定义指数窗函数 """
    pd = math.exp(-abs(x / width_a))
    return pd


def switch_win_func(win_type):
    """ 根据命令行参数确定窗函数类型 """
    if win_type == "uniform":
        return unifrom_win_func
    elif win_type == "gaussian":
        return gaussian_win_func
    elif win_type == "exponential":
        return exponential_win_func
    else:
        print("Please input a support window function type!")
        sys.exit(0)


def arg_turn2list(_arg, _arg_name):
    """ 判断参数合法性并解析多值参数转换为列表 """
    arg_list = _arg.split(",")
    if _arg_name == "num":
        for i in range(len(arg_list)):
            try:
                arg_list[i] = int(arg_list[i])
            except:
                print("[ERROR]\nThe number of samples value should be a integer")
                sys.exit(0)
    elif _arg_name == "width":
        for i in range(len(arg_list)):
            try:
                arg_list[i] = float(arg_list[i])
            except:
                print("[ERROR]\nThe window width value should be a float or integer")
                sys.exit(0)
    return arg_list


def num_list_average_pd(_list, win_wid, sample_sum):
    # 求每个点密度的平均
    _sum = 0
    for _ele in _list:
        _sum += _ele
    return _sum / (sample_sum * win_wid)


# ---------------------------------------------------------------
# main function
# ---------------------------------------------------------------


def main():
    """ main """
    args = parse_args()
    sample_number_arg_list = arg_turn2list(args.sample_number, "num")
    window_width_arg_list = arg_turn2list(args.window_width, "width")
    # 传入样本数和窗宽两个参数并分别转换为列表

    sample_len, width_len = len(
        sample_number_arg_list), len(window_width_arg_list)
    plt.figure(1, dpi=100)

    for i in range(sample_len):
        for j in range(width_len):  # 遍历所有样本数和窗宽组合
            sample_list = normal_random_gen(-1, 1, 1, 1,
                                            sample_number_arg_list[i])
            # 产生随机样本
            sample_point = iter(sample_list)
            pn_x = {}   # 使用字典存储每点概率密度
            window_function = switch_win_func(args.window_type)
            # 根据命令行参数确定窗函数类型

            pn_x = {}
            for x in sample_point:
                x_i = int(min(sample_list) - window_width_arg_list[j] / 2)
                step_length = 0.01
                end_point = int(max(sample_list) -
                                window_width_arg_list[j] / 2) + 1
                # 设定起始点和步长
                while x_i <= end_point:
                    if x_i in pn_x.keys():
                        pn_x[x_i].append(window_function(window_width_arg_list[j],
                                                         x_i - x))
                    else:
                        pn_x[x_i] = [window_function(window_width_arg_list[j],
                                                     x_i - x)]
                        # 将特定点在每个样本影响下的概率密度存为列表，密度的平均之后计算
                    x_i += step_length

            # 调用matplotlib绘图
            fig_x = pn_x.keys()
            fig_y = [num_list_average_pd(pn_x[_key], window_width_arg_list[j],
                                         sample_number_arg_list[i]) for _key in fig_x]
            # 计算每个点平均密度
            plt.subplot2grid((sample_len, width_len), (i, j))
            # 依次把每个参数组合画为子图
            plt.plot(fig_x, fig_y)
            if i == 0:
                plt.title("hn=%s" % str(round(window_width_arg_list[j], 2)),
                          fontsize=10)
            if j == 0:
                plt.ylabel("N=%d" % sample_number_arg_list[i],
                           fontsize=10)

    plt.suptitle("%s window function" % args.window_type)
    # 图标标题代表窗类型
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
