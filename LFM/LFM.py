# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     LFM
   email:         695492835@qq.com
   Author :       sjyttkl
   date：          2020/3/28
   Description :  https://blog.csdn.net/qq_35883464/article/details/90768402
==================================================
"""
__author__ = 'songdongdong'

import numpy as np
from util import  *
def lfm_train(train_data,F,alpha,beta,step):
    """
    LFM 训练脚本,不断更新 user_vec item_vec
    :param train_data: 训练数据
    :param F:  user vector len,item vector len
    :param alpha: regularization factor
    :param beta: leanring rate
    :param step: iteration num
    :return: dict: key: itemid,value:list
              dict key userid,value:list
    """
    user_vec = {}
    item_vec = {}
    for setp_index in range(step):
        for data_instance in train_data:
            user_id,item_id ,label = data_instance
            if user_id not in user_vec:
                user_vec[user_id] = init_model(F)
            if item_id not in item_vec:
                item_vec[item_id] = init_model(F)
        #模型迭代代码的书写
        delta = label  - model_predict(user_vec[user_id],item_vec[item_id])
        for index in range(F):# 在这里和公式不一样，是实际工程的应用，没有2倍
            user_vec[user_id][index] += beta*(delta*item_vec[item_id][index] - alpha*user_vec[user_id][index])
            item_vec[item_id][index] += beta*(delta*user_vec[user_id][index] - alpha*item_vec[item_id][index])
        beta = beta*0.9
    return user_vec,item_vec

def init_model(vector_len):
    """
    初始化参数
    Args:
    :param vector_len:
    :return:
    """
    return np.random.randn(vector_len)


def model_predict(user_vecotr,item_vector):
    """
    user_vecotr  and item_Vecotr distance
    Args:
    :param user_vecotr:  model produce user vector
    :param item_vector:  model proeuce item vector
    :return: user_vector 和 itemr_vector 距离，推荐程度。cos
    a num
    """
    res = np.dot(user_vecotr,item_vector)/(np.linalg.norm(user_vecotr)* np.linalg.norm(item_vector))
    return res

def model_train_process():
    """
    test lfm model train
    :return:
    """
    train_data = get_train_data("../data/ratings.txt")
    user_vec, item_vec =lfm_train(train_data,50,0.01,0.1,50)
    recom_result = give_recom_result(user_vec, item_vec, '11')
    ana_recom_result(train_data, '11', recom_result)

def give_recom_result(user_vec, item_vec, userid):
    '''
    user lfm model result give fix userid recom result
    :param user_vec: lfm model result
    :param item_vec: lfm model result
    :param userid: fix userid
    :return:list : [(itemid,score), (itemid1,score1)]
    '''
    if userid not in user_vec:
        return []
    record = {}
    recom_list = []
    fix_num = 5
    user_vecor = user_vec[userid]
    for itemid in item_vec:
        item_vector = item_vec[itemid]
        res = np.dot(user_vecor,item_vector)/(np.linalg.norm((user_vecor)*np.linalg.norm(item_vector)))
        record[itemid] = res                    # record={'itemid':'res'}
        record_list = list(record.items())      # record.items()=[(itemid,res),(itemid,res)]   在此用list转换格式是Python3的不兼容问题
    for zuhe in sorted(record.items(), key=lambda rec: record_list[1], reverse=True)[:fix_num]:
        itemid = zuhe[0]
        score = round(zuhe[1],3)
        recom_list.append((itemid,score))
    return recom_list



def ana_recom_result(train_data,userid,recom_list):
    '''
    :param train_data: 之前用户对哪些电影打分高
    :param userid: 分析的用户
    :param recom_list:模型给出的推荐结果
    '''
    item_info = get_item_info('../data/movies.txt')
    for data_instance in train_data:
        userid1,itemid,label = data_instance
        if userid1 == userid and label == 1:
            print(item_info[itemid])
    print('返回结果：')
    for zuhe in recom_list:
        print(item_info[zuhe[0]])


if __name__ == "__main__":
    model_train_process()