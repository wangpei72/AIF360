import math
import sys
import os

import pandas as pd

sys.path.append("../")
import numpy as np

data_set_list = ['adult', 'compas', 'german', 'bank',
                 'default', 'heart', 'student',
                 'home_credit']

csv_set_list = ['../../../data/raw/adult/adult.csv',
                '../../../data/raw/compas/compas-scores-two-years.csv',
                '../../../data/raw/german/german.csv',
                '../../../data/raw/bank/bank-additional-full.csv',
                '../../../data/raw/default/default_of_credit_card_clients.csv',
                '../../../data/raw/heart/processed.cleveland.data.csv',
                '../../../data/raw/student/Student.csv',
                '../../../data/raw/home/home.csv'
                ]

meps_set_list = ['meps15', 'meps16']
meps_csv_list = ['../../../data/raw/meps/h181.csv',
                 '../../../data/raw/meps/h192.csv']


def map_unique_as_num(df, colum_name):
    list_type = df[colum_name].unique()
    dic = dict.fromkeys(list_type)
    for i in range(len(list_type)):
        dic[list_type[i]] = i
    df[colum_name] = df[colum_name].map(dic)


def print_cols(df):
    cols = df.columns
    for i in cols:
        print(i)


def binary_numeric_by_param(df, col_name, param):
    df[col_name] = df[col_name].apply(lambda x: np.int(x > param))


def binary_numeric_by_mean(df, col_name):
    mean = df[col_name].mean()
    df[col_name] = df[col_name].apply(lambda x: np.int(x >= mean))


def age_normalize(df, col_name):
    df[col_name] = df[col_name].apply(lambda x: int(int(x) / 10))
    df.loc[df[col_name] < 1, col_name] = 1
    df.loc[df[col_name] > 9, col_name] = 9


def clip_numeric_normalize(df, col_name):
    normal_to_a_b(df, col_name, 1, 9)
    df.loc[df[col_name] < 1, col_name] = 1
    df.loc[df[col_name] > 9, col_name] = 9


def get_attr_str(df, dataset_name):
    cols = df.columns
    tar = dataset_name + ' :\n'
    cnt = 0
    for i in cols:
        tar += '\''
        tar += i
        tar += '\''
        if cnt % 10 == 0:
            tar += '\n'
        if cnt == len(cols) - 1:
            break
        else:
            tar += ','
        cnt += 1
    tar += '\n\n'
    return tar


def get_feat_str(df):
    cols = df.columns
    tar = ''
    cnt = 0
    for i in cols:
        tar += i
        if cnt % 10 == 0:
            tar += '\n'
        if cnt == len(cols) - 1:
            break
        else:
            tar += ','
        cnt += 1
    tar += '\n\n'
    return tar


def get_des_str(df, dataset_name, Y_feat, D_feat, Y_map, D_map, P_map):
    D_features = ['sex', 'age']
    Y_features = ['credit']
    label_maps = {1.0: 'Good Credit', 2.0: 'Bad Credit'}
    all_privileged_classes = {'sex': [1],
                              "age": lambda x: x > 2}
    all_protected_attribute_maps = {'sex': {1: 'Male', 0: 'Female'},
                                    "age": {1: 'Old', 0: 'Young'}}
    shape = df.values.shape
    test_shape = int(np.floor(0.2 * shape[0]))
    train_shape = int(np.floor(0.8 * shape[0]))
    tar = dataset_name + ':\nTotal shape is: '
    tar += str(shape) + '\n'
    tar += 'Train set shape is: '
    tar += '(' + str(train_shape) + ',' + str(shape[1]) + ')\n'
    tar += 'Test set shape is: '
    tar += '(' + str(test_shape) + ',' + str(shape[1]) + ')\n'
    tar += 'Features: '
    tar += get_feat_str(df)
    tar += 'Labels is : ' + Y_feat[0]
    tar += '\nProtected attrs : '
    for x in D_feat:
        tar += x + ' '
    tar += '\nMeta-data :\n'
    tar += 'label_map : ' + str(Y_map)
    tar += '\nall_protected_attribute_maps : ' + str(D_map)

    return tar


def wrt_descrip_txt(df, dataset_name, Y_feat, D_feat, Y_map, D_map, P_map):
    # target_dir = 'aif360/data/npy_data/' + dataset_name +'-aif360preproc/'
    # if not os._exists(target_dir):
    #     os.makedirs(target_dir)
    target_path = dataset_name + '-description.txt'
    description = get_des_str(df, dataset_name, Y_feat, D_feat, Y_map, D_map, P_map)
    with open(target_path, 'w+') as f:
        f.write(description)
        f.close()


def get_num_base(x):
    x = abs(x)
    n = int(math.log10(x))
    n_pow = int(math.pow(10, n))
    return n_pow


def normal_to_0_10(df, col_name):
    normal_to_a_b(df, col_name, 0, 10)
    # df[col_name] = df[col_name].apply(lambda x: int(int(x) / get_num_base(x)))
    df.loc[df[col_name] < 1, col_name] = 0
    df.loc[df[col_name] > 9, col_name] = 10


def normal_to_1_99(df, col_name):
    normal_to_a_b(df, col_name, 1, 99)
    df.loc[df[col_name] < 1, col_name] = 1
    df.loc[df[col_name] > 99, col_name] = 99


def normal_to_a_b(df, col_name, a, b):
    #     （1）首先找到样本数据Y的最小值Min及最大值Max
    # （2）计算系数为：k=（b-a)/(Max-Min)
    # （3）得到归一化到[a,b]区间的数据：norY=a+k(Y-Min)
    min = df[col_name].min()
    max = df[col_name].max()
    k = np.float(b - a) / np.float(max - min)
    df[col_name] = df[col_name].apply(lambda x: int(a + k * (x - min)))


def age_to_div_10(df, col_name):
    df[col_name] = df[col_name].apply(lambda x: np.int(x / 10))


def work_flow(df, y_labels, skip_feat=None, binary_0_feat=None,
              age_feat=None, binary_avg_feat=None, norm_1_99=None,
              norm_0_10=None, age_div_10=None):
    for i in df.columns:
        if i == 'Unnamed: 0':
            continue
        elif y_labels is not None and i in y_labels:
            continue
        elif skip_feat is not None and i in skip_feat:
            continue
        elif age_div_10 is not None and i in age_div_10:
            age_to_div_10(df, i)
        elif norm_1_99 is not None and i in norm_1_99:
            normal_to_1_99(df, i)
        elif norm_0_10 is not None and i in norm_0_10:
            normal_to_0_10(df, i)
        elif binary_0_feat is not None and i in binary_0_feat:
            binary_numeric_by_param(df, i, 0)
        elif binary_avg_feat is not None and i in binary_avg_feat:
            binary_numeric_by_mean(df, i)
        elif age_feat is not None and i in age_feat:
            age_normalize(df, i)
        elif df[i].dtype == np.dtype('int64'):
            clip_numeric_normalize(df, i)
        elif df[i].dtype == np.dtype('float64'):
            clip_numeric_normalize(df, i)
        elif df[i].dtype == np.dtype('O'):
            map_unique_as_num(df, i)
        else:
            print('none supported dtype in df')
    print('work flow done')
    return df


def wrt_atrrs():
    for i in range(len(data_set_list)):
        file_name = 'sep_all_attrs_for_8_dataset.txt'
        data_set_name = data_set_list[i]
        df = pd.read_csv(csv_set_list[i])
        with open(file_name, 'a+') as f:
            tar = get_attr_str(df, data_set_name)
            f.write(tar)
            f.close()
    for i in range(len(meps_set_list)):
        file_name = 'sep_all_attrs_for_meps_dataset.txt'
        data_set_name = meps_set_list[i]
        df = pd.read_csv(meps_csv_list[i])
        with open(file_name, 'a+') as f:
            tar = get_attr_str(df, data_set_name)
            f.write(tar)
            f.close()


if __name__ == '__main__':
    wrt_atrrs()
    # data_set_name = data_set_list[0]
    # csv_file = csv_set_list[0]
    # df = pd.read_csv(csv_file)
    # df = work_flow(df, 'income-per-year', ['sex', 'race', 'education-num'])

    # if df['native-country'].dtype == np.dtype('O'):
    #     print('yes')
    # if df['age'].dtype == np.dtype('float64'):
    #     print('age yes')
    # else:
    #     print('age no')
    # for col in df.columns:
    #     if col == 'Unnamed: 0':
    #         continue
    #     dtype = df[col].dtype
    #     print(col, dtype)
