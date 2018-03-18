#!/usr/bin/env python
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
    Python packages: argparse
"""

# ---------------------------------------------------------------
# import
# ---------------------------------------------------------------

import argparse

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
        required=True,
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

# ---------------------------------------------------------------
# main function
# ---------------------------------------------------------------


def main():
    """ main """
    args = parse_args()


if __name__ == "__main__":
    main()
