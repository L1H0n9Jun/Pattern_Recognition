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
    Li Hongjun

Description:
    context

Reurirements:
    Python packages: argparse, numpy, matplotlib
"""

# ---------------------------------------------------------------
# import
# ---------------------------------------------------------------

import sys
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
        type=int,
        required=True,
        help="""
        Number of samples used to estimate the population.
        """
    )
    parser.add_argument(
        '-w', '--window_width',
        type=float,
        required=True,
        help="""
        Parzen window width used.
        """
    )
    parser.add_argument(
        '-t', '--window_type',
        type=str,
        required=True,
        choices=['uniform', 'gaussian'],
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
        sample_list.append(0.2 * _rand_nor_1[i] + 0.8 * _rand_nor_2[i])
    sample_list.sort()
    return sample_list


def unifrom_win_func(width_a, x):
    """ 定义window function """
    if (-1.0 / 2 * width_a) <= x <= (1.0 / 2 * width_a):
        return 1.0 / width_a
    else:
        return 0


def gaussian_win_func(width_a, x):
    pass


def switch_win_func(win_type):
    if win_type == "uniform":
        return unifrom_win_func
    elif win_type == "gaussian":
        return gaussian_win_func
    else:
        print("Please input a support window function type!")
        sys.exit(0)

# ---------------------------------------------------------------
# main function
# ---------------------------------------------------------------


def main():
    """ main """
    args = parse_args()

    sample_list = normal_random_gen(-1, 1, 1, 1, args.sample_number)
    sample_point = iter(sample_list)
    pn_x = {}
    window_function = switch_win_func(args.window_type)
    for x in sample_point:
        pn_x[x] = 0
        for sample in sample_list:
            pn_x[x] += window_function(args.window_width, sample - x)
        pn_x[x] /= args.sample_number

    fig_x = pn_x.keys()
    fig_y = [pn_x[_key] for _key in fig_x]
    plt.figure(1, dpi=150)
    plt.plot(fig_x, fig_y)
    plt.show()


if __name__ == "__main__":
    main()
