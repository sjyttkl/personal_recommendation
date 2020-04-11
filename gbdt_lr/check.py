# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     check
   email:         695492835@qq.com
   Author :       sjyttkl
   date：          2020/4/8
   Description :  
==================================================
"""
__author__ = 'songdongdong'
import sys
import numpy as np
import xgboost as xgb
import utils
import train
from scipy.sparse import csc_matrix  #稀疏矩阵，方便计算
import math
def get_test_data(test_file, feature_num_file):
    """读取测试文件"""
    total_feature_num = 103
    # total_feature_num = utils.get_feature_num(feature_num_file)
    test_label = np.genfromtxt(test_file, dtype=np.float32, delimiter=',', usecols=-1)
    total_feature_list = range(total_feature_num)
    test_feature = np.genfromtxt(test_file, dtype=np.float32, delimiter=',', usecols=total_feature_list)

    return test_feature, test_label


def predict_by_tree(test_feature, tree_model):
    """predict by gbdt model"""
    predict_list = tree_model.predict(xgb.DMatrix(test_feature))
    return predict_list


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
    auc_score = (total_pos_index - (pos_num) * (pos_num + 1) / 2) / (pos_num * neg_num + 1)

    print("auc: %5f " % (auc_score))


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
    print("accuracy: %5f    " % (accuracy_score))


def run_check_core(test_feature, test_label, model, score_func):
    """评分函数"""
    predict_list = score_func(test_feature, model)
    get_auc(predict_list, test_label)
    get_accuracy(predict_list, test_label)


def run_check_xgb(test_file, tree_model_file, feature_num_file):
    """gbdt模型预测"""
    test_feature, test_label = get_test_data(test_file, feature_num_file)
    # 加载模型
    test_model = xgb.Booster(model_file=tree_model_file)
    # 打分函数
    run_check_core(test_feature, test_label, test_model, predict_by_tree)

def run_check_lr_gbdt_core(test_feature,test_label,mix_tree_mode,mix_lr_coef,tree_info,score_func):
    """

    :param test_feature: 测试特征
    :param test_label:  测试label
    :param mix_tree_mode:  混合树模型
    :param mix_lr_coef:   混合lr模型
    :param tree_info:    树信息（树量、数深度、学习率
    :param score_func:   打分函数
    :return:
    """
    predict_list = score_func(test_feature,mix_tree_mode,mix_lr_coef,tree_info)
    get_auc(predict_list,test_label)
    get_accuracy(predict_list,test_label)

def get_mix_model_tree_info():
    """tree info of mix model"""
    tree_depth = 4
    tree_num = 10
    step_size = 0.3
    result =(tree_depth,tree_num,step_size)
    return result

def predict_by_lr_gbdt(test_feature,mix_tree_model,mix_lr_coef,tree_info):
    """
    predict_by_lr_gbdt
    """
    #首先预测 每个样本在gbdt模型，落在哪个样本上
    tree_leaf = mix_tree_model.predict(xgb.DMatrix(test_feature),pred_leaf=True)
    #然后展开tree_info
    (tree_depth,tree_num,step_size) = tree_info
    # (tree_depth, tree_num, learning_rate) = get_mix_model_tree_info()

    total_feature_list = train.get_gbdt_and_lr_featrue(tree_leaf,tree_depth=tree_depth,tree_num=tree_num)


    result_list = np.dot(csc_matrix(mix_lr_coef),total_feature_list.tocsc().T).toarray()[0]

    sigmod_ufunc = np.frompyfunc(simoid,1,1)

    return sigmod_ufunc(result_list)

def simoid(x):
    return 1 / (1 + math.exp(-x))



def run_check_lr_gbdt(test_file, mix_tree_model_file, mix_lr_model_file, feature_num_file):
    """gbdt模型预测"""
    test_feature, test_label = get_test_data(test_file, feature_num_file)
    # 加载模型
    mix_tree_model = xgb.Booster(model_file=mix_tree_model_file)

    mix_lr_coef = np.genfromtxt(mix_lr_model_file, dtype=np.float32, delimiter=",")

    # tree_info = (4, 10, 0.3)  # 深度，颗数 总的步长
    tree_info= get_mix_model_tree_info()
    run_check_lr_gbdt_core(test_feature, test_label, mix_tree_model,
                           mix_lr_coef, tree_info, predict_by_lr_gbdt)

    # 打分函数
    run_check_core(test_feature, test_label, mix_tree_model, predict_by_tree)


if __name__ == "__main__":
    # if len(sys.argv) == 4:
    #     test_file = sys.argv[1]
    #     tree_model = sys.argv[2]
    #     feature_num_file = sys.argv[3]
    #     run_check(test_file, tree_model, feature_num_file)
    # elif len(sys.argv) == 5:
    #     test_file = sys.argv[1]
    #     tree_mix_model = sys.argv[2]
    #     lr_coef_mix_model = sys.argv[3]
    #     feature_num_file = sys.argv[4]
    #     run_check_lr_gbdt(test_file, tree_mix_model, lr_coef_mix_model,  feature_num_file)
    # else:
    #     print ("check gbdt model usage: python xx.py test_file  tree_model feature_num_file")
    #     print ("check lr_gbdt model usage: python xx.py test_file tree_mix_model lr_coef_mix_model feature_num_file")
    #     sys.exit()


    run_check_xgb("data/gbdt_test_file", "data/gbdt.model", "data/gbdt_feature_num", )
    run_check_lr_gbdt("data/gbdt_test_file","data/xgb_mix_model", "data/xgb_lr_coef_mix_model",
                      "data/gbdt_feature_num", )