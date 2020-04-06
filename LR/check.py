# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     check
   email:         695492835@qq.com
   Author :       sjyttkl
   date：          2020/4/1
   Description :  测试
==================================================
"""
from __future__ import division
import utils
__author__ = 'songdongdong'

# auc解释
# 1 0.9
# 1 0.8
# 1 0.3
# 0 0.2
# 0 0.4  3*2  2+2+1 => auc = 5/6;  工业界 auc到0.7以上就可以应用


import numpy as np
from sklearn.externals import joblib
import math


def get_test_data(test_file,feature_num_file):
    """读取测试文件"""
    total_feature_num = 118
    total_feature_num = utils.get_feature_num(feature_num_file)
    print(total_feature_num)
    test_label = np.genfromtxt(test_file, dtype=np.float32, delimiter=',', usecols=-1)
    total_feature_list = range(total_feature_num)
    test_feature = np.genfromtxt(test_file, dtype=np.float32, delimiter=',', usecols=total_feature_list)

    return test_feature, test_label


def predict_by_lr_model(test_feature, lr_model):
    """模型打分函数"""
    result_list = []
    prob_list = lr_model.predict_proba(test_feature)
    # print(prob_list[0])
    for index in range(0, len(prob_list)):
        result_list.append(prob_list[index][1])  # 将label为1的结果放到 list里返回
    return result_list


def predict_by_lr_coef(test_feature, lr_coef):
    """通过模型文件（进行预测"""
    sigmod_func = np.frompyfunc(simoid, 1, 1)  # 可以对array每个元素 进行相应的函数 操作，1,1，表示一个输入，一个输出
    return sigmod_func(np.dot(test_feature, lr_coef))


def simoid(x):
    return 1 / (1 + math.exp(-x))


def get_auc(predict_list, test_label):
    """
    auc得分
    :param predict_list: model predict score list
    :param test_label:  label of test data
    auc = sum(pos_index) - pos_num(pos_num+1)/2  / (pos_num *neg_num)

    """
    total_list = []
    for index in range(len(predict_list)):
        predict_score = predict_list[index]
        label = test_label[index]
        total_list.append((label, predict_score))
    scorted_total_list = sorted(total_list, key=lambda ele: ele[1])
    neg_num = 0
    pos_num = 0
    count = 1
    total_pos_index = 0

    for value in scorted_total_list:
        label, predict_score = value
        if label == 0:
            neg_num += 1
        else:
            pos_num += 1
            total_pos_index += count
        count += 1
    auc_score = (total_pos_index - (pos_num) * (pos_num + 1) / 2) / (pos_num * neg_num+1)

    print("auc: %5f "%(auc_score))


def get_accuracy(predict_list, test_label):
    """
    predict_list:model predict list score list
    test_label: lable of test data
    :return:
    """
    right_num = 0
    score_thr = 0.5
    for index in range(len(predict_list)):
        predict_score = predict_list[index]
        if predict_score >= score_thr:
            predict_label = 1
        else:
            predict_label = 0

        if predict_label == test_label[index]:
            right_num += 1
    total_num = len(predict_list)
    accuracy_score = right_num / total_num
    print("accuracy: %5f    "%(accuracy_score))


def run_check_score(test_feature, test_label, model, score_func):
    """打分函数"""

    predict_list = score_func(test_feature, model)
    get_auc(predict_list, test_label)
    get_accuracy(predict_list, test_label)


def run_check(test_file, lr_coef_file, lr_model_file,feature_num_file):
    test_feature, test_label = get_test_data(test_file,feature_num_file)
    lr_coef_file = np.genfromtxt(lr_coef_file, dtype=np.float32, delimiter=",")
    lr_model = joblib.load(lr_model_file)

    run_check_score(test_feature, test_label, lr_model, predict_by_lr_model)
    run_check_score(test_feature, test_label, lr_coef_file, predict_by_lr_coef)


if __name__ == "__main__":
    run_check("../data/test_file", "../data/lr_coef", "../data/lr_model_filejb","../data/feature_num")
