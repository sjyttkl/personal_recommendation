# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     ana_train.data
   email:         695492835@qq.com
   Author :       sjyttkl
   date：          2020/3/29
   Description :  树模型，不需要对 连续特征进行离散化，只需要对离散特征进行01编码，gbdt特征 的预处理阶段
==================================================
"""
__author__ = 'songdongdong'
import sys
import numpy as np
import pandas as pd


def get_input(input_train_file, input_test_file):
    dtype_dict = {"age": np.int32, "education-num": np.int32, "capital-gain": np.int32, "capital-loss": np.int32,
                  "hour-per-week": np.int32}
    use_list = [i for i in range(15)]
    use_list.remove(2)
    print(use_list)

    train_data_df = pd.read_csv(input_train_file, sep=', ', header=0, dtype=dtype_dict, na_values="?", usecols=use_list)
    train_data_df = train_data_df.dropna(axis=0, how="any")
    test_data_df = pd.read_csv(input_test_file, sep=', ', header=0, dtype=dtype_dict, na_values="?", usecols=use_list)
    test_data_df = test_data_df.dropna(axis=0, how="any")
    print(train_data_df.shape, test_data_df.shape)
    return train_data_df, test_data_df


def label_trans(x):
    if x.strip() == "<=50K":
        return "0"
    elif x.strip() == ">50K":
        return "1"
    else:
        return "0"


def process_label_feature(label_feature_str, df_in):
    """处理label"""
    df_in.loc[:, label_feature_str] = df_in.loc[:, "wage_class"].apply(label_trans)
    df_in = df_in.drop(columns=['wage_class'], axis=1)
    return df_in


def dis_to_feature(x, feature_dict):  # one-hot编码
    output_list = [0] * len(feature_dict)
    if x not in feature_dict:
        return ",".join([str(ele) for ele in output_list])  # 没有找到相应位置：[0,0,0,0,0,0]
    else:
        index = feature_dict[x]
        output_list[index] = 1
    return ",".join([str(ele) for ele in output_list])  # 找到相应位置[1,0,0,0,0]


def process_dis_feature(label_feature_str, df_train, df_test):
    """特征的离散化"""

    origin_dict = df_train.loc[:, label_feature_str].value_counts().to_dict()
    feature_dict = dict_trans(origin_dict)
    # print(feature_dict)
    df_train.loc[:, label_feature_str] = df_train.loc[:, label_feature_str].apply(dis_to_feature, args=(feature_dict,))
    df_test.loc[:, label_feature_str] = df_test.loc[:, label_feature_str].apply(dis_to_feature, args=(feature_dict,))
    # print(df_train.loc[:5, label_feature_str])
    return len(feature_dict)


def dict_trans(dict_in):
    output_dict = {}
    index2 = 0
    for index in sorted(dict_in.items(), key=lambda x: x[1], reverse=True):
        output_dict[index[0]] = index2
        index2 += 1
    return output_dict


def list_trans(input_dict):
    """
    连续特征的处理
    :param intput_dict: {'count': 30162.0,'mean': 38.437901995888865,'std': 13.134664776856338,'min': 17.0,'25%': 28.0,'50%': 37.0,'75%': 47.0,'max': 90.0}
    :return:[0.1,0.2,0.3,0.4,0.5]
    """
    output_list = [0] * 5
    key_list = ['min', "25%","50%","75%", "max"]
    for index in range(len(key_list)):
        fix_key = key_list[index]
        if fix_key not in input_dict:
            print("error-")
            sys.exit(1)
        else:
            output_list[index] = input_dict[fix_key]
    return output_list


def con_to_feature(x, feature_list):
    """连续特征值处理"""
    feature_len = len(feature_list) - 1
    result = [0] * feature_len
    for index in range(feature_len):
        if x > feature_list[index] and x <= feature_list[index + 1]:
            result[index] = 1
            return ",".join([str(ele) for ele in result])
    return ",".join([str(ele) for ele in result])


def process_con_feature(feature_str, df_train, df_test):
    orgin_dict = df_train.loc[:, feature_str].describe().to_dict()
    feature_list = list_trans(orgin_dict)
    df_train.loc[:, feature_str] = df_train.loc[:, feature_str].apply(con_to_feature, args=(feature_list,))
    df_test.loc[:, feature_str] = df_test.loc[:, feature_str].apply(con_to_feature, args=(feature_list,))

    # print(df_train.loc[:3, feature_str])
    # print(feature_list)
    return len(feature_list) - 1


def ana_train_data(input_trian_data, input_test_data, output_train_file, out_test_file, feature_num_file):
    train_data_df, test_data_df = get_input(input_trian_data, input_test_data)
    train_data_df = process_label_feature("label", train_data_df)
    test_data_df = process_label_feature("label", test_data_df)

    # 特征的离散化
    dis_feature_list = ['workclass', "education", "marital_status", "occupation", "relationship", "race", "sex",
                        "native_country"]
    con_feature_list = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']

    index_list = ["age", "workclass", "education", "education_num", "marital_status", "occupation", "relationship",
                  "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country"]

    dis_feature_num = 0  # 记录离散到多少维
    con_feature_num = 0  # 记录离散到多少维

    for dis_feature in dis_feature_list:
        tmp_feature_num = process_dis_feature(dis_feature, train_data_df, test_data_df)
        dis_feature_num += tmp_feature_num
    for con_feature in con_feature_list:
        # 连续性的特征，树模型不需要对连续特征进行处理,直接加1就行
        tmp_feature_num = process_con_feature(con_feature, train_data_df, test_data_df)
        con_feature_num += 1


    output_file(train_data_df, output_train_file)
    output_file(test_data_df, out_test_file)
    with open("../data/" + feature_num_file, "w", encoding="utf-8") as file_w:
        file_w.write("feature_num=" + str(dis_feature_num + con_feature_num))
    print(dis_feature_num)
    print(con_feature_num)


def output_file(data_df, output_file):
    with open(output_file, "w", encoding="utf-8") as file:
        for row_index in data_df.index:
            # print(row_index)
            # print([str(ele) for ele in data_df[row_index].values])
            outline = ",".join([str(ele) for ele in data_df.loc[row_index].values])
            file.write(outline + "\n")


def add(str_one, str_two):
    """

    :param str_one: 0,0,1,0
    :param str_two: 0,0,0,1
    :return: 0,0,1,0,0
    """
    list_one = str_one.strip().split(",")
    list_two = str_two.strip().split(",")

    list_one_len = len(list_one)
    list_two_len = len(list_two)

    return_list = [0] * (list_one_len * list_two_len)

    try:
        index_one = list_one.index("1")  # 找到1 的位置
    except:
        index_one = 0
    try:
        index_two = list_two.index("1")
    except:
        index_two = 0
    return_list[index_one * list_two_len + index_two] = 1
    return ",".join([str(ele) for ele in return_list])


def combine_feature(feature_one, feature_two, new_feature, train_data_df, test_data_df, feature_num_dict):
    """
    组合特征
    :param feature_one:
    :param feature_two:
    :param new_feature:  组合特征的name
    :param train_data_df:
    :param test_data_df:
    :param feature_num_dict: ndim of every feature,key feature name value len of the dim
    :return: new_feature_num 新特征的维度
    """

    train_data_df[new_feature] = train_data_df.apply(lambda row: add(row[feature_one], row[feature_two]), axis=1)
    test_data_df[new_feature] = test_data_df.apply(lambda row: add(row[feature_one], row[feature_two]), axis=1)

    if feature_one not in feature_num_dict:
        print("error")
        sys.exit()
    if feature_two not in feature_num_dict:
        print("error")
        sys.exit()

    return feature_num_dict[feature_one] * feature_num_dict[feature_two]


if __name__ == "__main__":
    ana_train_data("../data/adult.data", "../data/adult.test", "../data/gbdt_train_file", "../data/gbdt_test_file",
                   "../data/gbdt_feature_num")

    # 测试函数
    with open("../data/gbdt_train_file", "r", encoding="utf-8") as file:
        count = 0
        for line in file.readlines():
            item = line.strip().split(",")
            print(len(item))
            count += 1
            if count >= 10:
                break
