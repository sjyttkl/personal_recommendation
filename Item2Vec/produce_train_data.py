# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     produce_train_data
   email:         695492835@qq.com
   Author :       sjyttkl
   date：          2020/3/29
   Description :  
==================================================
"""
__author__ = 'songdongdong'
import sys
import os
def produce_train_data(input_file,out_file):
    """
    从用户的评分文件，获取训练文件
    :param input_file:  user behavior file
    :param out_file:
    :return:
    """
    if not os.path.exists(input_file):
        return
    record = {}
    score_thr=4
    with open(input_file ,"r",encoding="utf-8") as file:
        for line in file.readlines()[1:]:
            item = line.strip().split(",")
            if len(item) <4:
                continue
            user_id,item_id,rating = item[0],item[1],float(item[2])
            if rating <float(score_thr):
                continue
            if user_id not in record:
                record[user_id] = []
            record[user_id].append(item_id)
    with open(out_file,"w",encoding="utf-8") as w_file:
        for userid in record:
            w_file.write(" ".join(record[userid]) +"\n")

if __name__ == "__main__":
    if len(sys.argv)<3:
        print("userag :python xx.pyt inputfile outoutfile")
    else:
        inputfile = sys.argv[1]
        outputfile = sys.argv[1]
    produce_train_data("../data/ratings.txt","../data/train_data.txt")