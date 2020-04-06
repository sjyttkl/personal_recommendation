# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     train
   email:         songdongdong@jd.com
   Author :       songdongdong
   date：          2020/3/30
   Description :  
==================================================
"""
__author__ = 'songdongdong'
from sklearn.linear_model import LogisticRegressionCV as LRCV #带有交叉验证
import numpy as np
from sklearn.externals import joblib
import utils

def train_lr_mode(train_file,model_coef,model_file,feature_num_file):


    tolal_feature_num = 118
    tolal_feature_num = utils.get_feature_num(feature_num_file)
    train_label = np.genfromtxt(train_file,dtype=np.int32,delimiter=',',usecols= -1)
    feature_list = range(tolal_feature_num)
    train_feature = np.genfromtxt(train_file,dtype=np.int32,delimiter=",",usecols=feature_list)
    lr_cf = LRCV(Cs=[1],penalty="l2",tol=0.00001,max_iter=1000,cv=5).fit(train_feature,train_label)
    #CS 为 正则化参数 1 或者0.1， 0.01  ：[1,10,100] 的倒数
    #tol 参数迭代停止的条件：
    #cv=5 5折交叉验证
    #solver 梯度下降的方式:坐标轴下降法、拟牛顿法、随机梯度下降法（适应大数据量，随机抽取数据进行迭代）；因为是l2正则化，所以只能选择拟牛顿法和随机梯度下降；
           #默认是拟牛顿法

    scores = lr_cf.scores_.values()
    scores = list(scores)[0] #提取values值
    print(scores)
    print("diff %s : "%("   ".join([str(ele) for ele in scores.mean(axis=0)]))) #按照列求均值
    print("Accuracy %s ,(+- %0.2f ): "%(scores.mean(),scores.std()*2))

    #模型的auc   Cs=[1,10,100]
    lr_cf = LRCV(Cs=[1], penalty="l2", tol=0.00001, max_iter=1000, cv=5,scoring='roc_auc').fit(train_feature, train_label)
    scores = lr_cf.scores_.values()
    scores = list(scores)[0]  # 提取values值
    print(scores)
    print("diff %s : " % ("  ".join([str(ele) for ele in scores.mean(axis=0)])))  # 按照列求均值
    print("AUC %s ,(+- %0.2f ):"%(scores.mean(),scores.std()*2))
   # 发现 Cs=[1,10,100]  正则化参数  为 1的时候，效果最好，故而，后面全选1


    #保存模型文件
    coef = lr_cf.coef_[0]
    with open(model_coef,"w+",encoding='utf-8') as file:
        file.write(",".join(str(ele) for ele in coef))

    joblib.dump(lr_cf,model_file+"jb")


if __name__ == "__main__":
    train_lr_mode("../data/train_file","../data/lr_coef","../data/lr_model_file","../data/feature_num")