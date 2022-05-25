import math
import sys
import os
import re
import pandas as pd

from aif360.algorithms.preprocessing.optim_preproc_helpers import helper

sys.path.append("../")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.stats import chi2_contingency, kruskal, f_oneway, normaltest, bartlett
import numpy as np
from datetime import date

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
    # 进行计算之前应该将所有包含na会导致计算失败的行drop掉
    df.dropna(subset=[col_name], inplace=True)
    min = df[col_name].min()
    max = df[col_name].max()
    k = np.float(b - a) / np.float(max - min)
    df[col_name] = df[col_name].apply(lambda x: int(a + k * (x - min)))


def age_to_div_10(df, col_name):
    df[col_name] = df[col_name].apply(lambda x: np.int(x / 10))


def binary_by_a(df, col_name, a):
    df.loc[df[col_name] < a, col_name] = 1
    df.loc[df[col_name] > a, col_name] = 0
    df[col_name] = df[col_name].apply(lambda x: int(x))


def binary_by_4_5(df, col_name):
    binary_by_a(df, col_name, 4.5)


def binary_by_93(df, col_name):
    binary_by_a(df, col_name, 93)


def binary_equal_a(df, col_name, a):
    df.loc[df[col_name] == 1, col_name] = 0
    df.loc[df[col_name] == a, col_name] = 1
    df.loc[df[col_name] != 1, col_name] = 0
    df[col_name] = df[col_name].apply(lambda x: int(x))


def binary_equal_999(df, col_name):
    binary_equal_a(df, col_name, 999)


def binary_by_0(df, col_name):
    df.loc[df[col_name] < 0, col_name] = 0
    df.loc[df[col_name] > 0, col_name] = 1
    df[col_name] = df[col_name].apply(lambda x: int(x))


def normal_to_0_99(df, col_name):
    normal_to_a_b(df, col_name, 0, 99)
    df.loc[df[col_name] < 0, col_name] = 0
    df.loc[df[col_name] > 99, col_name] = 99

def normal_to_99_99(df, col_name):
    normal_to_a_b(df, col_name, -99, 99)


def do_c_days_from_compas(df, col_name):
    df.dropna(subset=[col_name], inplace=True)
    df.loc[df[col_name] <= 100, col_name] *= 0.09
    df.loc[df[col_name] > 100, col_name] = 10
    df[col_name] = df[col_name].apply(lambda x: np.int(x))


def do_days_b_screening_arrest(df, col_name):
    #     days_b_screening_arrest
    df.dropna(subset=[col_name], inplace=True)
    df.loc[df[col_name] == 0, col_name] = 0
    df.loc[df[col_name] > 0, col_name] = 1
    df.loc[(df[col_name] < 0) & (df[col_name] > -1), col_name] = -1
    df.loc[df[col_name] < -1, col_name] = -2
    df[col_name] = df[col_name].apply(lambda x: np.int(x))


def int_cast(df, col_name):
    df[col_name] = df[col_name].apply(lambda x: int(x))

def map_yes_or_no(df, col_name):
    df[col_name] = df[col_name].replace({'yes': 1, 'no': 0})

def work_flow(df, y_labels, skip_feat=None, binary_0_feat=None,
              age_feat=None, binary_avg_feat=None, norm_1_99=None,
              bin_0=None, bin_4_5=None, bin_999=None, bin_93=None, int_cas=None,
              norm_0_10=None, norm_0_9=None, age_div_10=None, norm_0_99=None,
              map_yes_no=None, norm_99_99=None,
              days=None, c_days_from_compas=None, days_b_screening_arrest=None):
    for i in df.columns:
        if i == 'Unnamed: 0':
            continue
        elif y_labels is not None and i in y_labels:
            continue
        elif skip_feat is not None and i in skip_feat:
            continue
        elif norm_99_99 is not None and i in norm_99_99:
            normal_to_99_99(df, i)
        elif map_yes_no is not None and i in map_yes_no:
            map_yes_or_no(df, i)
        elif int_cas is not None and i in int_cas:
            int_cast(df, i)
        elif bin_0 is not None and i in bin_0:
            binary_by_0(df, i)
        elif bin_4_5 is not None and i in bin_4_5:
            binary_by_4_5(df, i)
        elif bin_999 is not None and i in bin_999:
            binary_equal_999(df, i)
        elif bin_93 is not None and i in bin_93:
            binary_by_93(df, i)
        elif days_b_screening_arrest is not None and i in days_b_screening_arrest:
            do_days_b_screening_arrest(df, i)
        elif c_days_from_compas is not None and i in c_days_from_compas:
            do_c_days_from_compas(df, i)
        elif norm_0_9 is not None and i in norm_0_9:
            normal_to_a_b(df, i, 0, 9)
        elif days is not None and i in days:
            normal_to_0_99(df, i)
        elif age_div_10 is not None and i in age_div_10:
            age_to_div_10(df, i)
        elif norm_0_99 is not None and i in norm_0_99:
            normal_to_0_99(df, i)
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


if __name__ == '__main__':
    XD_features = ['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10'
    ,'X11','X12','X13','X14','X15','X16','X17','X18','X19','X20'
    ,'X21','X22','X23']
    drop = []
    d_features_in_x = ['X2']
    D_features = ['sex']
    X_features = list(set(XD_features) - set(drop)- set(d_features_in_x))
    Y_features = ['Y']

    all_privileged_classes = {'sex': [1]}
    all_protected_attribute_maps = {'X2': {1: 'Male', 0: 'Female'}}
    label_maps = {1: 'yes', 0: 'no'}

    df = pd.read_csv('../../../data/raw/meps/h181.csv', sep=',', header=[0])
    # df.dropna(inplace=True)
    # df.drop('Unnamed: 0',axis=1, inplace=True)
    # df['Y'] = df['Y'].replace({'yes': 1, 'no': 0})
    # df['X2'] = df['X2'].replace({1: 1, 2: 0})
    # # # 3 进行numeric归一化方式的判断，整理出list，传入workflow参数
    # # int_cas = []
    # age_div_10 = []
    # skip_feat = []
    # norm_99_99 = []
    # norm_0_99 = []
    # norm_1_99 = []
    # df = work_flow(df, y_labels=Y_features,  age_div_10=age_div_10,
    #                skip_feat=skip_feat, norm_0_99=norm_0_99,
    #                norm_99_99=norm_99_99, norm_1_99=norm_1_99)
    # helper.wrt_descrip_txt(df, 'default', Y_feat=Y_features, D_feat=D_features,
    #                        Y_map=label_maps, D_map=all_protected_attribute_maps,
    #                        P_map=all_privileged_classes)
    print('done')
