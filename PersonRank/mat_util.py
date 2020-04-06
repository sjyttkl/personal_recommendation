# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     mat_util
   email:         songdongdong@jd.com
   Author :       songdongdong
   date：          2020/3/29
   Description :  
==================================================
"""
__author__ = 'songdongdong'


from util import *
from scipy.sparse import coo_matrix  # 系数矩阵存储方式
import numpy as np


def graph_to_m(graph):
    '''
    由之前的二分图得到矩阵公式中的M矩阵，所有(item+user)顶点，所有(item+user)顶点位置（为了求r）
    :param graph: user and item graph-->   {userA:{itemb:1,itemc:1},itemb:{userA:1}}
    :return:
    matrix M,
    a list 所有(item+user)顶点,
    a dict 所有(item+user)顶点位置
    '''

    vertex = list(graph.keys())  # 所有(item+user)顶点
    address_dict = {}  # 所有(item+user)顶点位置
    total_len = len(vertex)
    for index in range(len(vertex)):
        address_dict[vertex[index]] = index  # 每一行对应一个顶点，每个顶点处于什么样的位置
    row = []
    col = []
    data = []
    for element in graph:  # element所有item+user的顶点
        weight = round(1 / len(graph[element]), 3)  # graph[element]二分图中所有和element相连接的顶点
        row_index = address_dict[element]
        for element in graph[element]:  #这里是获取到  与 element 相连的节点
            col_index = address_dict[element]  #列向量
            row.append(row_index)
            col.append(col_index)
            data.append(weight)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    m = coo_matrix((data, (row, col)), shape=(total_len, total_len)) #系数矩阵，
    return m, vertex, address_dict


def mat_all_point(m_matrix, vertex, alpha):
    '''
    矩阵算法personal_rank的公式，得到（1-alpha*M^T）
    矩阵算法personal_rank的公式
    :param m_matrix:
    :param vertex: 所有(item+user)顶点
    :param alpha: 随机游走的概率
    :return:  系数矩阵
    '''

    # 初始化单位矩阵（如果使用numpy创建，容易超内存）
    total_len = len(vertex)
    row = []
    col = []
    data = []
    for index in range(total_len):
        row.append(index)
        col.append(index)
        data.append(1)
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    eye_t = coo_matrix((data, (row, col)), shape=(total_len, total_len))
    # print(eye_t.todense())
    return eye_t.tocsr() - alpha * m_matrix.tocsr().transpose() #tocsr() 可以加速系数矩阵运算

if __name__ =="__main__":
    graph  = get_graph_from_data("../data/log.txt")
    m,vertex,address_dict  = graph_to_m(graph)
    mat_all_point(m,vertex,0.8)
    # print(address_dict)
    # print(m.todense())