# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     produce_item_sim.py
   email:         695492835@qq.com
   Author :       songdongdong
   date：          2020/3/29
   Description :  
==================================================
"""
__author__ = 'songdongdong'
import os
import numpy as np


# 通过训练文件得到 wordembedding文件后，进行读取embedding文件
def load_item_vec(input_file):
    if not os.path.exists(input_file):
        return {}
    item_vec = {}
    with open(input_file, "r", encoding="utf-8") as file:
        for line in file.readlines()[1:]:
            item = line.strip().split()
            if len(item) < 129:
                continue
            itemid = item[0]
            if itemid == "</s>":
                continue
            item_vec[itemid] = np.array([float(ele) for ele in item[1:]])

    return item_vec

def cal_item_sim(item_vec,itemid,output_file):
    if itemid not in item_vec:
        return
    score = {}
    topK =10
    fix_item_vec = item_vec[itemid]
    for tmp_itemid in item_vec:
        if tmp_itemid == itemid:
            continue
        tmp_itemvec = item_vec[tmp_itemid]
        fenmu = np.linalg.norm(fix_item_vec) * np.linalg.norm(tmp_itemvec) #两个向量摸的乘积：作为分母
        if fenmu ==0:
            score[tmp_itemid] = 0
        else :
            score [tmp_itemid] = round(np.dot(fix_item_vec,tmp_itemvec)/fenmu,3) #cos距离
    with open(output_file,"w+",encoding="utf-8")as file_w:
        out_str = itemid +"\t"
        tmp_list = []
        for zuhe in sorted(score.items(),key=lambda x:x[1],reverse=True)[:topK]:
            tmp_list.append(zuhe[0] +"_"+str(zuhe[1]))
        out_str += ";".join(tmp_list)
        file_w.write(out_str+"\n")

def run_main(input_file,output_file):
    item_vec = load_item_vec(input_file)
    cal_item_sim(item_vec,str(27),output_file)


if __name__ == "__main__":
    item_vec = load_item_vec("../data/item_vec.txt")
    print(len(item_vec))
    print(item_vec["318"])
    run_main("../data/item_vec.txt","../data/sim_result.txt")