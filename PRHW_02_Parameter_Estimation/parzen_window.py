#!/usr/bin/env python
# -*- coding: utf-8 -*-
from scipy.constants.constants import sigma

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
    Python packages: argparse, numpy
"""

# ---------------------------------------------------------------
# import
# ---------------------------------------------------------------

import argparse
import numpy as np

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
        '-i', '--input_file',
        type=str,
        required=False,
        help="""
        Transmit file inputted to the script.
        """
    )
    parser.add_argument(
        '-o', '--output_file',
        type=str,
        # required=True,
        default="default.output",
        help="""
        Define name of file outputted.
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
    return sample_list

# ---------------------------------------------------------------
# main function
# ---------------------------------------------------------------


def main():
    """ main """
    args = parse_args()


if __name__ == "__main__":
    main()
