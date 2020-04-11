# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     train.py
   email:         695492835@qq.com
   Author :       sjyttkl
   date：          2020/4/10
   Description :  
==================================================
"""
__author__ = 'songdongdong'

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import os
import tensorflow as tf

def get_featrue_column():
    """
    get wid featrue and deep feature
    :return:
    wide featrue colunm ,deep feature colunm
    #特征如下：
    age, workclass, fnlwgt, education, education_num, marital_status, occupation,
    relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country, wage_class
    """

    # 连续特征 --- 放到  deep层
    age = tf.feature_column.numeric_column("age")
    education_num = tf.feature_column.numeric_column("education_num")
    caption_gain = tf.feature_column.numeric_column("capital_gain")
    capital_loss = tf.feature_column.numeric_column("capital_loss")
    hours_per_week = tf.feature_column.numeric_column("hours_per_week")

    # 离散特征：首先进行哈希，哈希得到的特征放到wide层，然后再进行embedding 放入到deep层

    work_class = tf.feature_column.categorical_column_with_hash_bucket("workclass",
                                                                       hash_bucket_size=512)  # 因为离散特征取值总的数目，不会超过512
    education = tf.feature_column.categorical_column_with_hash_bucket("education",
                                                                      hash_bucket_size=512)  # 因为离散特征取值总的数目，不会超过512
    marital_status = tf.feature_column.categorical_column_with_hash_bucket("marital_status",
                                                                           hash_bucket_size=512)  # 因为离散特征取值总的数目，不会超过512
    occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation",
                                                                       hash_bucket_size=512)  # 因为离散特征取值总的数目，不会超过512
    relationship = tf.feature_column.categorical_column_with_hash_bucket("relationship",
                                                                         hash_bucket_size=512)  # 因为离散特征取值总的数目，不会超过512

    # 对年龄、收入、支出 连续特征进行离散化
    age_bucket = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    gain_bucket = tf.feature_column.bucketized_column(caption_gain, boundaries=[0, 1000, 2000, 3000, 10000])
    loss_bucket = tf.feature_column.bucketized_column(capital_loss, boundaries=[0, 1000, 2000, 3000, 5000])

    # 交叉特征---连续特征的交叉
    cross_columns = [
        tf.feature_column.crossed_column([age_bucket, gain_bucket], hash_bucket_size=36),  # 年龄是9端，收入是5端
        tf.feature_column.crossed_column([gain_bucket, loss_bucket], hash_bucket_size=16)
    ]

    # wide 层 的特征应该包括：哈希的特征，离散特征、交叉的特征
    base_columns = [work_class, education, marital_status, occupation,
                    relationship, age_bucket, gain_bucket, loss_bucket]

    wide_columns = base_columns + cross_columns

    # 连续特征 + 哈希值的embedding 维度 是9位，因为 2**9 = 512；所以他能涵盖  哈希范围
    deep_columns = [age, education_num, caption_gain, capital_loss, hours_per_week,
                    tf.feature_column.embedding_column(work_class, 9),
                    tf.feature_column.embedding_column(education, 9),
                    tf.feature_column.embedding_column(marital_status, 9),
                    tf.feature_column.embedding_column(occupation, 9),
                    tf.feature_column.embedding_column(relationship, 9)
                    ]

    return wide_columns, deep_columns


def build_model_estimator(wide_column, deep_column, model_folder):
    """
    Args:
        wide_column: wide feature
        deep_column: deep feature
        model_folder: origin model output folder
    Return:
        model_es, serving_input_fn
    """
    model_es = tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_folder,
        linear_feature_columns=wide_column,
        linear_optimizer=tf.train.FtrlOptimizer(0.1, l2_regularization_strength=1.0),  # 优化器
        dnn_feature_columns=deep_column,
        dnn_optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.1, l1_regularization_strength=0.001,
                                                        l2_regularization_strength=0.001),
        dnn_hidden_units=[128, 64, 32, 16]
        # 这里需要注意样本是否均衡
        # 1. deep特征一共：5+ 5*9 = 50
        # 第一层： 50* 128 = 6400
        # 第二层： 128*64 = 8192
        # 第三层：64*32 = 2048
        # 第四层：32*16 = 512
        # 一共：17152个参数  因为 特征对应样本的比：1:100,：所以   17152(特征):1715200(样本)，但是我们只有3万训练数据，所以需要对训练数据进行 重复采样57倍

    )
    feature_column = wide_column + deep_column
    #
    feature_spec = tf.feature_column.make_parse_example_spec(feature_column)
    # 构建接收函数，
    serving_input_fn = (tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec))
    return model_es, serving_input_fn
def get_auc(predict_list, test_label):
    """
    Args:
        predict_list: model predict score list
        test_label: label of  test data
    auc = (sum(pos_index)-pos_num(pos_num + 1)/2)/pos_num*neg_num
    """
    total_list = []
    for index in range(len(predict_list)):
        predict_score = predict_list[index]
        label = test_label[index]
        total_list.append((label, predict_score))
    sorted_total_list = sorted(total_list, key = lambda ele:ele[1])
    neg_num = 0
    pos_num = 0
    count = 1
    total_pos_index = 0
    for zuhe in sorted_total_list:
        label, predict_score = zuhe
        if label == 0:
            neg_num += 1
        else:
            pos_num += 1
            total_pos_index += count
        count += 1
    auc_score = (total_pos_index - (pos_num)*(pos_num + 1)/2) / (pos_num*neg_num)
    print ("auc:%.5f" %(auc_score))


def input_fn(data_file, re_time, shuffle, batch_num, predict):
    """
    Args:
        data_file:input data , train_data, test_data
        re_time:time to repeat the data file
        shuffle: shuffle or not [true or false]
        batch_num:
        predict: train or test [true or false]
    Return:
        train_feature, train_label or test_feature
    """
    _CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                            [0], [0], [0], [''], ['']]

    _CSV_COLUMNS = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'gender',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
        'label'
    ]

    def parse_csv(value):
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('label')
        classes = tf.equal(labels, '>50K')
        return features, classes

    # 预测阶段，主协议返回feature,不需要返回label
    def parse_csv_predict(value):
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('label')
        return features

    # 读取训练数据，过滤第一行（标题），并过滤特殊字符
    data_set = tf.data.TextLineDataset(data_file).skip(1).filter(
        lambda line: tf.not_equal(tf.strings.regex_full_match(line, ".*\?.*"), True))
    if shuffle:
        data_set = data_set.shuffle(buffer_size=30000)
    if predict:
        data_set = data_set.map(parse_csv_predict, num_parallel_calls=5)
    else:
        data_set = data_set.map(parse_csv, num_parallel_calls=5)  # 产生数据

    data_set = data_set.repeat(re_time)
    data_set = data_set.batch(batch_num)
    return data_set



def train_wd_model(model_es, train_file, test_file, model_export_folder, serving_input_fn):
    """
    Args:
        model_es: wd estimator
        train_file:
        test_file:
        model_export_folder: model export for tf serving
        serving_input_fn: function for model export
    """
    total_run = 6
    for index in range(total_run):
        model_es.train(input_fn=lambda: input_fn(train_file, 10, True, 100, False))
        print(model_es.evaluate(input_fn=lambda: input_fn(test_file, 1, False, 100, False)))  # 评估 api
    model_es.export_savedmodel(model_export_folder, serving_input_fn)  # 输出模型,这里是提交给 tf-serving 搭建服务使用的

def get_test_label(test_file):
    """get label of test_file"""
    if not os.path.exists(test_file):
        return []
    fp = open(test_file,"r",encoding="utf-8")
    linenum = 0
    test_label_list = []
    for line in fp:
        if linenum == 0:
            linenum +=1
            continue
        if "?" in line.strip():
            continue
        item = line.strip().split(",")
        label_str = item[-1]
        if label_str == ">50K":
            test_label_list.append(1)
        elif label_str == "<=50K":
            test_label_list.append(0)
        else:
            print("error")
    fp.close()

def test_model_performance(model_es,test_file):
    """
    test model auc in test data
    :param model_es:
    :param test_file:
    :return:
    """
    test_label = get_test_label(test_file)
    result = model_es.predict(input_fn = lambda :input_fn(test_file,1,False,100,True))
    predict_list = []
    for one_res in result:
        if "probabilities" in one_res:
            predict_list.append(one_res['probabilities'][1])
    get_auc(predict_list,test_label)

def run_main(train_file, test_file, model_folder, model_export_folder):
    """
    Args:
        train_file:
        test_file:
        model_folder: origin model floder to put train model
        model_export_folder: for tf serving
    """
    wide_column, deep_column = get_featrue_column()
    model_es, serving_input_fn = build_model_estimator(wide_column, deep_column, model_folder)
    print(model_es,serving_input_fn)
    train_wd_model(model_es, train_file, test_file, model_export_folder, serving_input_fn)
    test_model_performance(model_es, test_file)


if __name__ == "__main__":
    run_main("../data/train.txt", "../data/test.txt", "../data/wd", "../data/wd_export")
