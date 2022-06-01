import math
import sys
import os
import re
import pandas as pd

from . import helper

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
    # df = df[df[col_name] != -1] # 缺值的列不参与计算
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


def classify_num_cols_before_workflow(df, y_labels, age_div_10=None, skip_feat=None, sex=None):
    norm_0_99 = []
    norm_0_9 = []
    years_cols = []
    skip_cols_besides = []
    str_cols = []

    for i in df.columns:
        if i == 'Unnamed: 0':
            continue
        elif y_labels is not None and i in y_labels:
            continue
        elif age_div_10 is not None and i in age_div_10:
            continue
        elif skip_feat is not None and i in skip_feat:
            continue
        elif sex is not None and i in sex:
            continue
        elif df[i].dtype == np.dtype('O'):
            str_cols.append(i)
        elif df[i].dtype == np.dtype('int64'):
            # df = df[df[i] != -1]
            # df_to_drop = df.loc[df[i] == -1, i]
            # df.drop(df_to_drop, axis=0, inplace=True)
            if ((df[i]==2014).sum() > 30000 or
                (df[i]==2015).sum() > 30000 or
                (df[i]==2016).sum() > 30000):
                years_cols.append(i)
            elif df[i].mean() > 100:
                norm_0_99.append(i)
            elif df[i].mean() > 1.5:
                norm_0_9.append(i)
            else:
                skip_cols_besides.append(i)
        elif df[i].dtype == np.dtype('float64'):
            # df = df[df[i] != -1]
            # df_to_drop = df.loc[df[i] == -1, i]
            # df.drop(df_to_drop, axis=0, inplace=True)
            if df[i].mean() > 100:
                norm_0_99.append(i)
            elif df[i].mean() > 1.5:
                norm_0_9.append(i)
            else:
                df[i] = df[i].apply(lambda x: int(x))
                skip_cols_besides.append(i)
        else:
            print('none supported dtype in df')
    print('done with classifying cols')
    total_len = len(norm_0_99) + len(norm_0_9) + len(years_cols) + len(skip_cols_besides) + len(str_cols)
    return norm_0_99, norm_0_9, years_cols, skip_cols_besides, str_cols

def do_years_cols(df, col_name):
    df[col_name] = df[col_name].apply(lambda x: int(abs(x - 2014)))


def work_flow(df, y_labels, skip_feat=None, binary_0_feat=None,
              age_feat=None, binary_avg_feat=None, norm_1_99=None,
              bin_0=None, bin_4_5=None, bin_999=None, bin_93=None, int_cas=None,
              norm_0_10=None, norm_0_9=None, age_div_10=None, norm_0_99=None,
              map_yes_no=None, norm_99_99=None, years_cols=None,
              days=None, c_days_from_compas=None, days_b_screening_arrest=None):
    for i in df.columns:
        if i == 'Unnamed: 0':
            continue
        elif y_labels is not None and i in y_labels:
            continue
        elif skip_feat is not None and i in skip_feat:
            continue
        elif years_cols is not None and i in years_cols:
            do_years_cols(df, i)
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

def get_cols_str(cols):
    cnt = 0
    tar = str(len(cols)) + '\n'
    for i in cols:
        tar += '\''
        tar += i
        tar += '\''
        if i != len(cols) - 1:
            tar += ','
        if cnt % 15 == 0:
            tar += '\n'
        cnt += 1
    return tar

def wtr_cols_txt(cols, title='col_names are: \n', file_name='meps15_data_attrs'):
    with open(file_name + '.txt', 'w+') as f:
        tar_str = get_cols_str(cols)
        f.write(title)
        f.write(tar_str)
        f.close()


if __name__ == '__main__':
    df = pd.read_csv('../../../data/raw/meps/h181.csv', sep=',', header=[0])
    cols_all_initial = df.columns
    print('initial cols number: %d' % len(cols_all_initial))
    print(cols_all_initial)
    wtr_cols_txt(cols_all_initial, title='meps15_initial_cols are:\n',
                 file_name='meps15_initial_cols')
    all_privileged_classes = {'RACE': [1]}
    all_protected_attribute_maps = {'RACE': {1: 'White', 0: 'Non-white'}}
    label_maps = {1: 'yes', 0: 'no'}


    def race(row):
        if ((row['HISPANX'] == 2) and (
                row['RACEV2X'] == 1)):  # non-Hispanic Whites are marked as WHITE; all others as NON-WHITE
            return 'White'
        return 'Non-White'


    df['RACEV2X'] = df.apply(lambda row: race(row), axis=1)
    df = df.rename(columns={'RACEV2X': 'RACE'})
    df = df.rename(columns={'FTSTU53X': 'FTSTU', 'ACTDTY53': 'ACTDTY', 'HONRDC53': 'HONRDC', 'RTHLTH53': 'RTHLTH',
                            'MNHLTH53': 'MNHLTH', 'CHBRON53': 'CHBRON', 'JTPAIN53': 'JTPAIN', 'PREGNT53': 'PREGNT',
                            'WLKLIM53': 'WLKLIM', 'ACTLIM53': 'ACTLIM', 'SOCLIM53': 'SOCLIM', 'COGLIM53': 'COGLIM',
                            'EMPST53': 'EMPST', 'REGION53': 'REGION', 'MARRY53X': 'MARRY', 'AGE53X': 'AGE',
                            'POVCAT15': 'POVCAT',
                            'INSCOV15': 'INSCOV'})  # for all other categorical features, remove values < -1

    df = df[df['REGION'] >= 0]  # remove values -1
    df = df[df['AGE'] >= 0]  # remove values -1
    df = df[df['MARRY'] >= 0]  # remove values -1, -7, -8, -9
    df = df[df['ASTHDX'] >= 0]  # remove values -1, -7, -8, -9
    df = df[(df[['FTSTU', 'ACTDTY', 'HONRDC', 'RTHLTH', 'MNHLTH', 'HIBPDX', 'CHDDX', 'ANGIDX', 'EDUCYR', 'HIDEG',
                 'MIDX', 'OHRTDX', 'STRKDX', 'EMPHDX', 'CHBRON', 'CHOLDX', 'CANCERDX', 'DIABDX',
                 'JTPAIN', 'ARTHDX', 'ARTHTYPE', 'ASTHDX', 'ADHDADDX', 'PREGNT', 'WLKLIM',
                 'ACTLIM', 'SOCLIM', 'COGLIM', 'DFHEAR42', 'DFSEE42', 'ADSMOK42',
                 'PHQ242', 'EMPST', 'POVCAT', 'INSCOV']] >= -1).all(
        1)]  # for all other categorical features, remove values < -1


    def utilization(row):
        return row['OBTOTV15'] + row['OPTOTV15'] + row['ERTOT15'] + row['IPNGTD15'] + row['HHTOTD15']

    df['RACE'] = df['RACE'].replace({'White': 1, 'Non-White': 0})
    df['TOTEXP15'] = df.apply(lambda row: utilization(row), axis=1)
    lessE = df['TOTEXP15'] < 10.0
    df.loc[lessE, 'TOTEXP15'] = 0
    moreE = df['TOTEXP15'] >= 10.0
    df.loc[moreE, 'TOTEXP15'] = 1
    df = df.rename(columns={'TOTEXP15': 'UTILIZATION'})
    df['SEX'] = df['SEX'].replace({1: 1, 2: 0})

    subset_age = ['AGE31X', 'AGE42X', 'AGE53X', 'AGE15X', 'AGELAST', 'DOBMM', 'DOBYY']
    subset_y =['OBTOTV15', 'OPTOTV15', 'ERTOT15', 'IPNGTD15', 'HHTOTD15']
    subset_race = ['RACEV1X', 'RACEV2X', 'RACEAX', 'RACEBX', 'RACEWX', 'RACETHX', 'HISPANX', 'HISPNCAT']
    drop = ['AGE31X', 'AGE42X',  'AGE15X', 'AGELAST', 'DOBMM', 'DOBYY', # AGE53X被选作AGE列，其余列drop
            'OBTOTV15', 'OPTOTV15', 'ERTOT15', 'IPNGTD15', 'HHTOTD15', # 这是计算Y的5个列，获得y之后drop
            'RACEV1X', 'RACEAX', 'RACEBX', 'RACEWX', 'RACETHX', 'HISPANX', 'HISPNCAT',  # 判断race的列，获得了race之后drop
            'DUID', 'PID', 'DUPERSID', 'PANEL' ,'FAMID31',
            'FAMID42','FAMID53','FAMID15','FAMIDYR','CPSFAMID',
            'MOPID31X', 'MOPID42X', 'MOPID53X', 'DAPID31X', 'DAPID42X', 'DAPID53X','HIEUIDX'
            # ID 个人特定标识的列，drop
            ]
    df.drop(drop, axis=1, inplace=True)

    d_features_in_x = []
    Y_features = ['UTILIZATION']
    D_features = ['RACE']
    features_to_keep = ['REGION', 'AGE', 'SEX', 'RACE', 'MARRY',
                        'FTSTU', 'ACTDTY', 'HONRDC', 'RTHLTH', 'MNHLTH', 'HIBPDX', 'CHDDX', 'ANGIDX',
                        'MIDX', 'OHRTDX', 'STRKDX', 'EMPHDX', 'CHBRON', 'CHOLDX', 'CANCERDX', 'DIABDX',
                        'JTPAIN', 'ARTHDX', 'ARTHTYPE', 'ASTHDX', 'ADHDADDX', 'PREGNT', 'WLKLIM',
                        'ACTLIM', 'SOCLIM', 'COGLIM', 'DFHEAR42', 'DFSEE42', 'ADSMOK42', 'PCS42',
                        'MCS42', 'K6SUM42', 'PHQ242', 'EMPST', 'POVCAT', 'INSCOV', 'UTILIZATION', 'PERWT15F']

    features_to_keep = features_to_keep or df.columns.tolist()
    keep = (set(features_to_keep) | set(D_features)
            | set(Y_features))
    df = df[sorted(keep - set(drop), key=df.columns.get_loc)]

    XD_features = df.columns
    print('XD_features and df.cols after first-stage preproc are%d'%len(XD_features))
    print(XD_features)
    wtr_cols_txt(XD_features, title='meps15_after_pre_cols are:\n',
                 file_name='meps15_after_pre_cols')

    X_features = list(set(XD_features) - set(drop) - set(d_features_in_x))
    age_div_10 = ['AGE']


    norm_0_99, norm_0_9, years_cols, skip_cols_besides, str_cols  = classify_num_cols_before_workflow(
        df, y_labels=Y_features, age_div_10=age_div_10, sex=['SEX'])
    skip_feat = ['SEX', 'REGION', 'MARRY', 'ACTDTY', 'HONRDC', 'RTHLTH',
                 'MNHLTH', 'ASTHDX', 'WLKLIM', 'ACTLIM', 'SOCLIM', 'DFHEAR42', 'DFSEE42',
                 'POVCAT', 'INSCOV']
    wtr_cols_txt(norm_0_9,'0-9 cols are\n', '0-9_norm')
    wtr_cols_txt(norm_0_99, '0-99 cols are\n', '0-99_norm')
    wtr_cols_txt(years_cols, 'years cols are \n', 'years_cols')
    wtr_cols_txt(list(set(skip_cols_besides) | set(skip_feat)), 'skip cols including params are \n', 'skip besides')
    wtr_cols_txt(str_cols, 'str cols are \n', 'str cols')






    df = work_flow(df, y_labels=Y_features,  age_div_10=age_div_10,
                   skip_feat=list(set(skip_cols_besides) | set(skip_feat)), norm_0_99=norm_0_99,
                    norm_0_9=norm_0_9, years_cols=years_cols)
    df.to_csv('df_keep_as_360_meps15_181.csv')
    helper.wrt_descrip_txt(df, 'meps15', Y_feat=Y_features, D_feat=D_features,
                           Y_map=label_maps, D_map=all_protected_attribute_maps,
                           P_map=all_privileged_classes)
    print('done')
