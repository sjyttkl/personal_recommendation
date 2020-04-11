#-*-coding:utf8-*-
"""
==================================================
   File Name：     ana_train.data
   email:         695492835@qq.com
   Author :       sjyttkl
   date：          2020/3/29
   Description :  https://blog.csdn.net/sjyttkl/article/details/105456063
==================================================
"""
from __future__ import division
import sys

sys.path.append("util")
import CF.util.reader as reader
import math
import operator

def transfer_user_click(user_click):
    """
    get item by user_click, {item_id:[user_id,user_id2]}
    Args:
        user_click: key userid, value:[itemid1, itemid2]
    Return:
        dict, key itemid value:[userid1, userid2]
    """
    item_click_by_user = {}
    for user in user_click:
        item_list = user_click[user]
        for itemid in item_list:
            item_click_by_user.setdefault(itemid, [])
            item_click_by_user[itemid].append(user)
    return  item_click_by_user


def base_contribution_score():
    """
    base usercf  user contirbution score

    """
    return 1


def update_contribution_score(item_user_click_count):
    """
    usercf user contribution score update v1
    Args:
        item_user_click_count: how many user have clicked this item
    Return:
        contribution score
    """
    return 1/math.log10(1 + item_user_click_count)


def update_two_contribution_score(click_time_one, click_time_two):
    """
    user cf user contribution score update v2
    Args:
         differrent user action time to the same item, click_time_one,click_time_two
    Return:
        contribution score
    """
    delta_time = abs(click_time_two -click_time_one)
    norm_num = 60*60*24
    delta_time = delta_time/ norm_num
    return 1/(1+delta_time)

def cal_user_sim(item_click_by_user, user_click_time):
    """
    get user sim info，主要是通过商品的点击数和  间隔时间，来判断两个用户的相似度
    Args:
        item_click_by_user: dict , key:itemid value:[userid1, userid2]
    Return:
        dict , key itemid , value: dict , value_key: userid_itemid value_value:simscore
    """
    co_appear = {}
    user_click_count = {}
    for itemid ,user_list in item_click_by_user.items():
        for index_i in range(0, len(user_list)):
            user_i = user_list[index_i]
            user_click_count.setdefault(user_i, 0)
            user_click_count[user_i] += 1 #这里保存的是每个用户点击次数
            if user_i + "_" + itemid not in user_click_time:
                click_time_one = 0
            else:
                click_time_one = user_click_time[user_i + "_" + itemid] #获取用户点击这个item的时间戳
            for index_j in range(index_i + 1, len(user_list)):
                user_j = user_list[index_j]
                if user_j + "_" + itemid not in user_click_time:
                    click_time_two = 0
                else:
                    click_time_two = user_click_time[user_j + "_" + itemid] #获取第二个用户点击这个商品的item
                co_appear.setdefault(user_i, {})
                co_appear[user_i].setdefault(user_j, 0)
                co_appear[user_i][user_j] += update_two_contribution_score(click_time_one, click_time_two)
                co_appear.setdefault(user_j, {})
                co_appear[user_j].setdefault(user_i, 0)
                co_appear[user_j][user_i] += update_two_contribution_score(click_time_one, click_time_two)
                #co_appear 保存的是两个用户对同一个商品的 点击时间差
    user_sim_info = {}
    user_sim_info_sorted = {}
    for user_i, relate_user in co_appear.items():
        user_sim_info.setdefault(user_i, {})
        for user_j, cotime in relate_user.items():
            user_sim_info[user_i].setdefault(user_j, 0)
            user_sim_info[user_i][user_j] = cotime/math.sqrt(user_click_count[user_i]*user_click_count[user_j])
            # cotime/(math.sqrt(user_id1 * user_id2)) ，计算用户之间的相似度
    for user in user_sim_info:# 对每个用户 相似的用户进行 排序
        user_sim_info_sorted[user] = sorted(user_sim_info[user].items(), key = operator.itemgetter(1), reverse=True)
    return user_sim_info_sorted


def cal_recom_result(user_click, user_sim):
    """
    recom by usercf algo，通过user_sim得到的 相似用户以及得分，去从 用户点击行为找到相关item并返回
    Args:
        user_click: dict, key userid , value [itemid1, itemid2]
        user_sim: key:userid value:[(useridj, score1),(ueridk, score2)]
    Return:
        dict, key userid value:dict value_key:itemid , value_value:recom_score
    """
    recom_result ={}
    topk_user = 3
    item_num = 5
    for user, item_list in user_click.items():
        tmp_dict = {}
        for itemid in item_list:
            tmp_dict.setdefault(itemid, 1)#临时存储 item出现次数
        recom_result.setdefault(user, {})
        for zuhe in user_sim[user][:topk_user]:
            userid_j, sim_score = zuhe
            if userid_j not in user_click:
                continue
            for itemid_j in user_click[userid_j][:item_num]:
                recom_result[user].setdefault(itemid_j, sim_score)
    return recom_result


def debug_user_sim(user_sim):
    """
    print user sim result
    Args:
        user_sim: key userid value:[(userid1, score1), (userid2, score2)]
    """
    topk = 5
    fix_user = "1"
    if fix_user not in user_sim:
        print ("invalid user")
        return
    for zuhe in user_sim[fix_user][:topk]:
        userid, score = zuhe
        print (fix_user + "\tsim_user" + userid + "\t" + str(score))


def debug_recom_result(item_info, recom_result):
    """
    print recom result for user
    Args:
        item_info: key itemid value:[title, genres]
        recom_result: key userid value dict , value key:itemid value value:recom_score
    """
    fix_user = "1"
    if fix_user not in recom_result:
        print ("invalid user for recoming result")
        return
    for itemid in recom_result["1"]:
        if itemid not in item_info:
            continue
        recom_score = recom_result["1"][itemid]
        print ("recom_result:" + ",".join(item_info[itemid]) + "\t" + str(recom_score))


def main_flow():
    """
    main flow
    """
    #通过用户行为，去找到相似的用户群体
    user_click, user_click_time = reader.get_user_click("../data/ratings.txt")
    item_info = reader.get_item_info("../data/movies.txt")
    item_click_by_user = transfer_user_click(user_click)
    user_sim = cal_user_sim(item_click_by_user, user_click_time)
    debug_user_sim(user_sim)

    #这里是通过  相似用户的点击，召回相关商品
    recom_result = cal_recom_result(user_click, user_sim)
    print (recom_result["1"])
    debug_recom_result(item_info, recom_result)

if __name__ == "__main__":
    main_flow()
