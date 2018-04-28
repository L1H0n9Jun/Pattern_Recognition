#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------------------------------------------------
# description
# ---------------------------------------------------------------

"""
Decision_tree.py
============
Please type "./Decision_tree.py -h" for usage help
    
Author:
    Li Hongjun

Description:
    This is a python3 script for relization of Decision Tree.

Reurirements:
    Python packages: scipy, numpy
"""

# ---------------------------------------------------------------
# import
# ---------------------------------------------------------------

import sys
sys.setrecursionlimit(10000)
from math import log
import scipy.io as sio
import numpy as np
import json

# ---------------------------------------------------------------
# function definition
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# main function
# ---------------------------------------------------------------


def main():
    """ main """
    data = sio.loadmat('Sogou_webpage.mat')
    label = np.transpose(data['doclabel'])[0]
    feature = data['wordMat']

    train_data, train_label, validate_data, validate_label, test_data, test_label = SplitData(
        feature, label)

    print("The impurity of raw data set is: %.6f" % Impurity(label))

    # 生成树
    DTree = GenerateTree(train_data, train_label)
    print("Origin tree gets %d / %d correct on train data => acuracy: %.6f\n"
          % CalAccuracy(DTree, train_data, train_label),
          "%d / %d correct on validate data => acuracy: %.6f\n"
          % CalAccuracy(DTree, validate_data, validate_label),
          "%d / %d correct on test data => acuracy: %.6f\n"
          % CalAccuracy(DTree, test_data, test_label),
          )

    # merge包含训练过程生成的"NULL"节点
    for i in range(10):
        Prune_NULL(DTree)

    print("Purned tree gets %d / %d correct on train data => acuracy: %.6f\n"
          % CalAccuracy(DTree, train_data, train_label),
          "%d / %d correct on validate data => acuracy: %.6f\n"
          % CalAccuracy(DTree, validate_data, validate_label),
          "%d / %d correct on test data => acuracy: %.6f\n"
          % CalAccuracy(DTree, test_data, test_label),
          )

    # 训练好的树写入文件存储
    with open("DTreeTrained.json", "w+") as f_out:
        f_out.write(json.dumps(DTree))


def GenerateTree(data, label, feat_pool=list(range(1200))):
    if label.size == 0:
        ''' 终止条件1 '''
        return 'NULL'

    if label.size == label.tolist().count(label[0]):
        ''' 终止条件2 '''
        return label[0]

    if len(feat_pool) == 0:
        ''' 终止条件3 '''
        label_count = {}
        for _lab in label:
            if _lab not in label_count.keys():
                label_count[_lab] = 0
            label_count[_lab] += 1

        max_lab = -1
        for _key in label_count.keys():
            if max_lab < label_count[_key]:
                max_lab = label_count[_key]
                max_key = _key
        return max_key

    best_feat = SelectFeature(data, label)
    cur_feat = feat_pool[best_feat]

    DTree = {cur_feat: {}}
    del(feat_pool[best_feat])              # 从feat_pool中删除当前使用的feature
    samp_idx = SplitNode(data, best_feat)  # 当前分支下两支各自的样本
    data_0, label_0, data_1, label_1 = Idx2Data(
        data, label, samp_idx, best_feat)

    # 进行递归建树
    DTree[cur_feat]["0"] = GenerateTree(data_0, label_0)
    DTree[cur_feat]["1"] = GenerateTree(data_1, label_1)
    return DTree


def SplitNode(samplesUnderThisNode, split_feat_idx):
    ''' 根据样本第feat_idx维的特征对样本进行分类，即树的分支 '''
    idx_0 = []
    idx_1 = []
    n = len(samplesUnderThisNode)
    for samp_idx in range(n):
        if samplesUnderThisNode[samp_idx][split_feat_idx] == 1:
            # 树的分支操作
            idx_1.append(samp_idx)
        else:
            idx_0.append(samp_idx)
    return idx_0, idx_1


def SelectFeature(samplesUnderThisNode, label):
    num_feat = len(samplesUnderThisNode[0])
    num_samp = len(label)
    base_entropy = Impurity(label)
    best_gain = -1.0

    # 遍历所有特征计算熵增
    for feat_idx in range(num_feat):
        cur_entropy = 0
        idx_0, idx_1 = SplitNode(samplesUnderThisNode, feat_idx)
        prob_0 = len(idx_0) / num_samp
        prob_1 = len(idx_1) / num_samp

        # 计算使用当前特征时的熵
        cur_entropy += prob_0 * Impurity(label[idx_0])
        cur_entropy += prob_1 * Impurity(label[idx_1])

        # 计算不纯度减少量
        info_gain = base_entropy - cur_entropy
        if info_gain > best_gain:
            best_gain = info_gain
            best_idx = feat_idx
    return best_idx


def Impurity(samples):
    # 使用熵来进行
    num_sample = samples.size
    sample_count = {}
    for _label in samples:
        if _label not in sample_count:
            sample_count[_label] = 0.0
        sample_count[_label] += 1
    entropy = 0.0
    for _key in sample_count:
        px_i = sample_count[_key] / num_sample
        entropy -= px_i * log(px_i, 2)
    return entropy


def Decision(GeneratedTree, XToBePredicted):
    ''' 递归进行树的遍历从而进行分类 '''
    if type(GeneratedTree).__name__ != 'dict':
        # 遍历至叶子节点，获得分类结果
        return GeneratedTree

    feat_idx = list(GeneratedTree.keys())[0]
    # 根节点特征索引
    nextbranch = GeneratedTree[feat_idx]

    if XToBePredicted[feat_idx] == 0:
        nextbranch = nextbranch["0"]
    else:
        nextbranch = nextbranch["1"]

    result = Decision(nextbranch, XToBePredicted)
    return result


def Prune_NULL(GeneratedTree):
    feat_root = list(GeneratedTree.keys())[0]
    second_dict = GeneratedTree[feat_root]

    for _key in second_dict.keys():
        if type(second_dict[_key]).__name__ == 'dict':
            # 未到达根节点，继续向下遍历树
            cur_root_node = list(second_dict[_key].keys())[0]
            for sub_key in second_dict[_key][cur_root_node].keys():
                if second_dict[_key][cur_root_node][sub_key] == 'NULL':
                    second_dict[_key] = second_dict[_key][cur_root_node][str(
                        1 - int(sub_key))]

            if type(second_dict[_key]).__name__ == 'dict':
                Prune_NULL(second_dict[_key])


def Idx2Data(samplesUnderThisNode, label, samp_idx, split_feat_idx):
    '''
    samp_idx: 分支操作获得的结果，包含两维，分别为各自分支的样本
    split_feat_idx 为当前分支所使用的特征
    '''
    idx_0 = samp_idx[0]
    idx_1 = samp_idx[1]
    data_0 = []
    data_1 = []

    # 获得分支后两支各自的样本特征和标签
    for i in idx_0:
        data_0.append(np.append(samplesUnderThisNode[i][:split_feat_idx],
                                samplesUnderThisNode[i][split_feat_idx + 1:]))
    for i in idx_1:
        data_1.append(np.append(samplesUnderThisNode[i][:split_feat_idx],
                                samplesUnderThisNode[i][split_feat_idx + 1:]))
    label_0 = label[idx_0]
    label_1 = label[idx_1]

    return data_0, label_0, data_1, label_1


def SplitData(data, label):
    train_data = []
    train_label = []
    validate_data = []
    validate_label = []
    test_data = []
    test_label = []
    idx = 0
    tmp_data = data.tolist()
    tmp_label = label.tolist()
    for i in range(1, 10):
        """ 按分层抽样原则进行抽样 """
        num_sample = len(label[label == i])
        step_len = int(num_sample * 3 / 5)
        num_left = num_sample - step_len

        # 每类的3/5作为测试集
        train_data += tmp_data[idx:idx + step_len]
        train_label += tmp_label[idx:idx + step_len]
        idx += step_len
        step_len = int(num_sample / 5)
        num_left -= step_len

        # 每类的1/5作为验证集
        validate_data += tmp_data[idx:idx + step_len]
        validate_label += tmp_label[idx:idx + step_len]
        idx += step_len

        # 每类剩下的1/5作为测试集
        test_data += tmp_data[idx:idx + num_left]
        test_label += tmp_label[idx:idx + num_left]
        idx += num_left
    return (np.array(train_data), np.array(train_label).astype(str), np.array(validate_data),
            np.array(validate_label).astype(str), np.array(test_data), np.array(test_label).astype(str))


def CalAccuracy(GeneratedTree, XToBePredicted, X_label):
    '''计算不同数据集上的准确率'''

    results = []
    num_correct = 0.0
    num_test = len(XToBePredicted)

    for ele in XToBePredicted:
        results.append(Decision(GeneratedTree, ele))

    for i in range(len(X_label)):
        if results[i] == X_label[i]:
            num_correct += 1
    accuracy = num_correct / num_test
    return num_correct, num_test, accuracy


if __name__ == "__main__":
    main()
