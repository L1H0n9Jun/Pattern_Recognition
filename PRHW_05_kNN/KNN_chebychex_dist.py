#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------------------------------------------------
# description
# ---------------------------------------------------------------

"""
KNN.py
============
Please type "./KNN.py -h" for usage help
    
Author:
    Li Hongjun

Description:
    A python3 script for the realization of KNN with minst data set.

Reurirements:
    Python packages: argparse, gzip, pickle (built-in), numpy
"""

# ---------------------------------------------------------------
# import
# ---------------------------------------------------------------

import time
import psutil
import os
import argparse
import gzip
import pickle as p
import numpy as np

# ---------------------------------------------------------------
# function definition
# ---------------------------------------------------------------


def parse_args():
    """ master argument parser """
    parser = argparse.ArgumentParser(
        description="A python3 script for the realization of KNN with minst data set.",
        # epilog="",
        # formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '-k', '--k_num',
        type=int,
        required=True,
        help="""
        The k argu for kNN.
        """
    )
    parser.add_argument(
        '-n', '--num_train',
        type=int,
        required=True,
        help="""
        The number of samples .
        """
    )

    args = parser.parse_args()
    return args


def load_data(data_name):
    ''' 直接从gz读取数据集，并将其序列化便于处理 '''
    with gzip.open(data_name, 'rb') as data_file:
        data_cont = p.load(data_file, encoding='latin1')
    return data_cont


def comp_dist(test_sample, train_sample):
    ''' 计算切比雪夫距离 '''
    test_num = test_sample.shape[0]
    train_num = train_sample.shape[0]
    dist = np.zeros((test_num, train_num))
    for i in range(test_num):
        test_img = test_sample[i, :]
        dist[i, :] = np.max(np.absolute(test_img - train_sample), axis=1)
    return dist


def pred_label(test_sample, train_sample, train_label, d=2, k=2):
    ''' 根据定义的距离和k值对测试集进行分类 '''
    dist = comp_dist(test_sample, train_sample)
    num_test = dist.shape[0]
    label_pred = np.zeros(num_test)
    for i in range(num_test):
        NearestNeighbor = []
        NearestNeighbor = train_label[np.argsort(dist[i, :])][:k]
        label_pred[i] = np.argmax(np.bincount(NearestNeighbor))
    return label_pred


# ---------------------------------------------------------------
# main function
# ---------------------------------------------------------------


def main():
    """ main """

    args = parse_args()

    # 读取数据集，数据集第一维为样本，第二维为样本标签
    # 其中train_set包含50000维（样本），test_set包含10000万维（样本）
    # 每维（样本）又由包含28*28=784 个元素（像素）的向量组成
    train_set = load_data('mnist.pkl.gz')[0]
    test_set = load_data('mnist.pkl.gz')[2]

    train_num = args.num_train  # <50000
    train_sample = train_set[0][range(train_num)]
    train_label = train_set[1][range(train_num)]

    test_num = 2000  # <10000
    test_sample = test_set[0][range(test_num)]
    test_label = test_set[1][range(test_num)]

    start = time.clock()

    # 测试样本分类
    test_label_pred = pred_label(test_sample, train_sample, train_label,
                                 k=args.k_num)
    correct_num = np.sum(test_label_pred == test_label)
    # 统计预测结果和真实label相同（正确）的样本数

    accuracy = float(correct_num) / test_num
    print('%d / %d with correct labels, accuracy: %.4f' %
          (correct_num, test_num, accuracy))

    end = time.clock()
    print("Time used: %.2f" % (end - start))
    print('Memory used: %d' % psutil.Process(os.getpid()).memory_info().rss)


if __name__ == "__main__":
    main()
