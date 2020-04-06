# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     utils
   email:         695492835@qq.com
   Author :       sjyttkl
   date：          2020/4/6
   Description :  
==================================================
"""
__author__ = 'songdongdong'

import os


def get_feature_num(feature_num_file):
    """
    获得特征数
    :param feature_num_file:
    :return:
    """
    if not os.path.exists(feature_num_file):
        return 0
    else:
        fp = open(feature_num_file, "r")
        for line in fp:
            item = line.strip().split("=")
            if item[0] == "feature_num":
                return int(item[1])
