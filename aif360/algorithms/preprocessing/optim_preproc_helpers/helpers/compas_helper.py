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
    min = df[col_name].min()
    max = df[col_name].max()
    k = np.float(b - a) / np.float(max - min)
    df[col_name] = df[col_name].apply(lambda x: int(a + k * (x - min)))


def age_to_div_10(df, col_name):
    df[col_name] = df[col_name].apply(lambda x: np.int(x / 10))


def normal_to_0_99(df, col_name):
    normal_to_a_b(df, col_name, 0, 99)
    df.loc[df[col_name] < 0, col_name] = 0
    df.loc[df[col_name] > 99, col_name] = 99


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
    df.loc[(df[col_name]<0) & (df[col_name]>-1), col_name] = -1
    df.loc[df[col_name] < -1, col_name] = -2
    df[col_name] = df[col_name].apply(lambda x: np.int(x))


def work_flow(df, y_labels, skip_feat=None, binary_0_feat=None,
              age_feat=None, binary_avg_feat=None, norm_1_99=None,
              norm_0_10=None, norm_0_9=None, age_div_10=None, norm_0_99=None,
              days=None,  c_days_from_compas=None, days_b_screening_arrest=None):
    for i in df.columns:
        if i == 'Unnamed: 0':
            continue
        elif y_labels is not None and i in y_labels:
            continue
        elif skip_feat is not None and i in skip_feat:
            continue
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
    XD_features = [
        'name', 'first', 'last', 'sex', 'dob', 'age', 'age_cat', 'race', 'juv_fel_count'
        , 'decile_score', 'juv_misd_count', 'juv_other_count', 'priors_count', 'days_b_screening_arrest', 'c_jail_in',
        'c_jail_out', 'c_case_number', 'c_offense_date', 'c_arrest_date'
        , 'c_days_from_compas', 'c_charge_degree', 'c_charge_desc', 'is_recid', 'r_case_number', 'r_charge_degree',
        'r_days_from_arrest', 'r_offense_date', 'r_charge_desc', 'r_jail_in'
        , 'r_jail_out', 'violent_recid', 'is_violent_recid', 'vr_case_number', 'vr_charge_degree', 'vr_offense_date',
        'vr_charge_desc', 'type_of_assessment', 'decile_score.1', 'score_text'
        , 'screening_date', 'v_type_of_assessment', 'v_decile_score', 'v_score_text', 'v_screening_date', 'in_custody',
        'out_custody', 'priors_count.1', 'start', 'end'
        , 'event'
    ]
    drop = ['id', 'name', 'first', 'last', 'compas_screening_date', 'r_case_number', 'dob',
            'age_cat', 'c_case_number', 'c_offense_date', 'c_arrest_date', 'r_charge_degree',
            'r_case_number', 'r_days_from_arrest',
            'r_offense_date', 'violent_recid', 'vr_case_number',
            'vr_charge_degree', 'vr_offense_date', 'vr_charge_desc', 'screening_date',
            'v_screening_date', 'v_type_of_assessment', 'type_of_assessment','priors_count.1','decile_score.1']
    X_feat = list(set(XD_features) - set(drop))
    D_features = ['sex', 'race']
    Y_features = ['two_year_recid']  # 优势为label ： 0
    all_privileged_classes = {"sex": [1],
                              "race": [1]}
    # protected attribute maps
    all_protected_attribute_maps = {"sex": {0: 'Male', 1: 'Female'},
                                    "race": {1: 'Caucasian', 0: 'Not Caucasian'}}
    label_maps = {1.0: 'Did recid.', 0.0: 'No recid.'}  # 优势label是 no-recid 无再犯
    features_to_drop = ['compas_screening_date' ]

    df = pd.read_csv('../../../data/raw/compas/compas-scores-two-years.csv')
    df.drop(['id', 'name', 'first', 'last', 'compas_screening_date', 'r_case_number', 'dob',
                  'age_cat', 'c_case_number', 'c_offense_date', 'c_arrest_date', 'r_charge_degree',
                   'r_case_number', 'r_days_from_arrest',
                  'r_offense_date', 'violent_recid', 'vr_case_number',
                  'vr_charge_degree', 'vr_offense_date', 'vr_charge_desc', 'screening_date',
                  'v_screening_date', 'v_type_of_assessment', 'type_of_assessment',
                  'priors_count.1','decile_score.1' ], axis=1, inplace=True)
    # df = df.dropna()
    # 进行日期的计算之前将所有日期都是na的行drop掉
    df.dropna(subset =['out_custody', 'in_custody',
                  'r_jail_in', 'r_jail_out',
                  'c_jail_in', 'c_jail_out'], how='all', inplace=True)

    newly_added_days_feat = ['length_of_stay', 'length_of_stay.1', 'length_of_stay.2']
    df['length_of_stay'] = (pd.to_datetime(df['c_jail_out']) -
                            pd.to_datetime(df['c_jail_in'])).apply(
        lambda x: x.days)
    df['length_of_stay.1'] = (pd.to_datetime(df['r_jail_out']) -
                              pd.to_datetime(df['r_jail_in'])).apply(
        lambda x: x.days)
    df['length_of_stay.2'] = (pd.to_datetime(df['out_custody']) -
                              pd.to_datetime(df['in_custody'])).apply(
        lambda x: x.days)
    df['start_end'] = df['start'].astype('int32') - df['end'].astype('int32')
    df['max_duration'] = df[['length_of_stay', 'length_of_stay.1', 'length_of_stay.2']].max(axis=1)
    # df = do_description(df)
    df.drop(['out_custody', 'in_custody',
                  'r_jail_in', 'r_jail_out',
                  'c_jail_in', 'c_jail_out',
                  'start', 'end',
                  'length_of_stay', 'length_of_stay.1', 'length_of_stay.2'], axis=1, inplace=True)
    df['sex'] = df['sex'].replace({'Female': 1, 'Male': 0})
    df['race'] = df['race'].apply(lambda x: np.int(x == 'Caucasian'))
    skip_feat = ['sex', 'race', 'juv_fel_count', 'decile_score', 'juv_misd_count', 'juv_other_count',
                 'priors_count', 'is_recid', 'is_violent_recid', 'decile_score.1', 'v_decile_score',
                 'priors_count.1', 'event']
    norm_0_99 = ['start', 'end', 'max_duration', 'start_end']
    norm_0_9 = []
    c_days = ['c_days_from_compas']
    days_b = ['days_b_screening_arrest']
    age_div_10 = ['age']

    df = work_flow(df, y_labels=Y_features, skip_feat=skip_feat, age_div_10=age_div_10,
                   norm_0_99=norm_0_99, norm_0_9=norm_0_9, days=newly_added_days_feat, c_days_from_compas=c_days,
                   days_b_screening_arrest=days_b)
    df.to_csv('df_compas.csv')
    helper.wrt_descrip_txt(df, 'compas', Y_feat=Y_features, D_feat=D_features, Y_map=label_maps
                           , D_map=all_protected_attribute_maps, P_map=all_privileged_classes)
    print('done')
