# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     train
   email:         695492835@qq.com
   Author :       sjyttkl
   date：          2020/4/6
   Description :  Practical Lessons from Predicting Clicks on Ads at Facebook (2014)论文阅读
==================================================
"""
__author__ = 'songdongdong'

import xgboost as xgb
import utils
import numpy as np
from sklearn.linear_model import  LogisticRegressionCV as LRCV
from scipy.sparse import coo_matrix
import sys
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
    # print(xgb.cv(params=para_dict, dtrain=train_mat, num_boost_round=tree_num, nfold=5, metrics={"auc"}))
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
    total_node_num  = 2**(tree_depth +1) -1
    leaf_num = 2**tree_depth
    not_leaf_num = total_node_num - leaf_num

    total_col_num = leaf_num * tree_num #总叶子节点数,总维度
    total_row_num = len(tree_leaf) #多少样本（
    col=[]
    row= []
    data  = []

    base_row_index = 0
    for one_result in tree_leaf:
        base_col_index = 0
        for fix_index in one_result:
            leaf_index = fix_index - not_leaf_num
            leaf_index = leaf_index if leaf_index >=0 else 0
            col.append(base_col_index +leaf_index)
            row.append(base_row_index)
            data.append(1)
            base_col_index += leaf_num
        base_row_index +=1
    total_feature_list = coo_matrix((data,(row,col)),shape=(total_row_num,total_col_num))

    return total_feature_list


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
    # 这里树的深度由 6 改为4，原因：如下：  深度为6：总共：127个节点，64个叶子节点，63个非叶子节点
    # 1.训练出的label,没有落在叶子节点上（或者落在叶子节点上比较少）
    # 2. 特征与样本量的比值：1:100。 因为： 10颗数，深度为6，则叶子节点有 有640个维度，而样本有3万条，不满足
    #训练树模型的代码
    bst = train_tree_model_core(train_mat,tree_depth,tree_num,learning_rate)
    bst.save_model(mix_tree_model_file)
    tree_leaf = bst.predict(train_mat,pred_leaf=True) #预测最终结果落在哪一个叶子节点上
    # print(tree_leaf) #[81 84 84 84 85 77 68 91 97 61] 代表10颗数，81代表最终1 落到那一颗叶子节点上
    # print(np.max(tree_leaf))
    # sys.exit()
    total_feature_list = get_gbdt_and_lr_featrue(tree_leaf,tree_num,tree_depth)

    #逻辑回归
    lr_clf = LRCV(Cs=[1.0], penalty="l2", tol=0.00001, max_iter=1000, cv=5, scoring='roc_auc').fit(total_feature_list,
                                                                                                train_label)
    scores = lr_clf.scores_.values()
    scores = list(scores)[0]  # 提取values值
    print(scores)
    print("diff %s : " % ("  ".join([str(ele) for ele in scores.mean(axis=0)])))  # 按照列求均值
    print("AUC %s ,(+- %0.2f ):" % (scores.mean(), scores.std() * 2))
    coef = lr_clf.coef_[0]
    with open(mix_lr_model_file,"w+",encoding="utf-8") as file:
        file.write(",".join([str(ele) for ele in coef]))




if __name__ == "__main__":
    # train_tree_model("../data/gbdt_train_file", "../data/gbdt_feature_num", "../data/gbdt.model")
    train_tree_and_lr_model("../data/gbdt_train_file", "../data/gbdt_feature_num", "../data/xgb_mix_model","../data/xgb_lr_coef_mix_model")
