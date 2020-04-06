# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     train
   email:         695492835@qq.com
   Author :       sjyttkl
   date：          2020/4/6
   Description :  
==================================================
"""
__author__ = 'songdongdong'

import xgboost as xgb
import utils
import numpy as np
from sklearn.linear_model import  LogisticRegressionCV as LRC

def get_train_data(train_file, feature_num_file):
    """获得训练数据"""
    total_feature_num = utils.get_feature_num(feature_num_file)
    train_label = np.genfromtxt(train_file, dtype=np.int32, delimiter=",", usecols=-1)
    feature_list = range(0, total_feature_num)
    train_feature = np.genfromtxt(train_file, dtype=np.int32, delimiter=",", usecols=feature_list)

    return train_feature, train_label


def train_tree_model_core(train_mat, tree_depth, tree_num, learning_rate):
    """

    :param train_mat: 训练数据和label
    :param tree_depth: 树的深度
    :param tree_num:树的数量
    :param learning_rate: 学习率
    :return:Booster
    """
    para_dict = {"max_depth": tree_depth, "eta": learning_rate, "objective": "reg:linear", "silent": 1}
    bst = xgb.train(params=para_dict, dtrain=train_mat, num_boost_round=tree_num)  # 交叉验证
    print(xgb.cv(params=para_dict, dtrain=train_mat, num_boost_round=tree_num, nfold=5, metrics={"auc"}))
    return bst


def choose_parameter():
    """生成参数"""
    result_list = []
    tree_depth_list = [4]  # [4,5,6]
    tree_num_list = [10, ]  # [10,50,100]
    learning_rate_list = [0.3, ]  # [0.3,,]
    for ele_tree_depth in tree_depth_list:
        for ele_tree_num in tree_num_list:
            for ele_learning_rate in learning_rate_list:
                result_list.append((ele_tree_depth, ele_tree_num, ele_learning_rate))

    return result_list


def grid_search(train_mat, ):
    """网格搜索参数"""
    para_list = choose_parameter()
    for ele in para_list:
        (tree_depth, tree_num, learning_rate) = ele
        para_dict = {"max_depth": tree_depth, "eta": learning_rate, "objective": "reg:linear", "silent": 1}
        res = xgb.cv(params=para_dict, dtrain=train_mat, num_boost_round=tree_num, nfold=5, metrics={"auc"})
        auc_score = res.loc[tree_num - 1, ["test-auc-mean"]].values[0]  # 获取最后一棵树的得分就行了。
        print("tree depth: %s , tree_num: %s , learning_rate: %s , auc: %f" % (
            tree_depth, tree_num, learning_rate, auc_score))


def train_tree_model(train_file, feature_num_file, tree_model_file):
    """
        训练树模型
    """
    train_feature, train_label = get_train_data(train_file, feature_num_file)
    train_mat = xgb.DMatrix(train_feature, train_label)
    # grid_search(train_mat) 参数选择，选择一次，就行，后面就无需继续使用了。
    tree_num = 10
    tree_depth = 6
    learning_rate = 0.3
    bst = train_tree_model_core(train_mat, tree_depth, tree_num, learning_rate)

    bst.save_model(tree_model_file)

def get_gbdt_and_lr_featrue(tree_leaf,tree_num,tree_depth):
    """
    提取特征的代码
    :param tree_leaf: predict of tree model
    :param tree_num: total_num
    :param tree_depth:
    :return: 返回稀疏矩阵: 因为树的深度越深，叶子节点就越多，再加上树很多。那叶子节点就更稀疏
    """





def train_tree_and_lr_model(train_file, feature_num_file, mix_tree_model_file, mix_lr_model_file):
    """
    gbdt+lr  混合模型  ，分开 训练 顺序训练（耗时较长）
    :param train_file: 训练数据
    :param feature_num_file:  特征维度文件
    :param mix_tree_model_file:  树模型文件
    :param mix_lr_model_file:   逻辑回归文件
    :return:  None
    """

    train_feature, train_label = get_train_data(train_file, feature_num_file)
    train_mat = xgb.DMatrix(train_feature, train_label)
    tree_num, tree_depth, learning_rate = 10,6, 0.3
    #训练树模型的代码
    bst = train_tree_model_core(train_mat,tree_depth,tree_num,learning_rate)
    bst.save_model(mix_tree_model_file)
    tree_leaf = bst.predcit(train_mat,pred_leaf=True) #预测最终结果落在哪一个叶子节点上
    total_feature_list = get_gbdt_and_lr_featrue(tree_leaf,tree_num,tree_depth)

    #逻辑回归


    pass


if __name__ == "__main__":
    train_tree_model("../data/gbdt_train_file", "../data/gbdt_feature_num", "../data/gbdt.model")
