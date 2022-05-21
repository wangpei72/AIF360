import math
import sys

sys.path.append("../")
import numpy as np
import pandas as pd

def map_unique_as_num(colum_name, df):
    list_type = df[colum_name].unique()
    dic = dict.fromkeys(list_type)
    for i in range(len(list_type)):
        dic[list_type[i]] = i
    df[colum_name] = df[colum_name].map(dic)

def binary_numeric_by_mean(df, col_name):
    mean = df[col_name].mean()
    df[col_name] = df[col_name].apply(lambda x: np.int(x >= mean))

def clip_numeric_normalize(df, col_name):
    mean = df[col_name].mean()
    mean_abs = abs(mean)
    n = int(math.log10(mean_abs))
    n_pow = int(math.pow(10, n))

    df[col_name] = df[col_name].apply(lambda x: int(x)/ n_pow)
    df.loc[df[col_name] < 1, col_name] = 1
    df.loc[df[col_name] > 9, col_name] = 9

def get_base(x):
    x_ab = abs(x)
    n = int(math.log10(x_ab))
    n_pow = int(math.pow(10, n))
    return n_pow

data_set_list = ['adult', 'compas', 'german', 'bank',
                 'default', 'heart', 'student', 'meps',
                 'home_credit']

# data_shape_list = [(None, 18), (None, 10), (None, 11)]
data_path_list = ['../data/npy_data_from_aif360/adult-aif360preproc/',
                  '../data/npy_data_from_aif360/compas-aif360preproc/',
                  '../data/npy_data_from_aif360/german-aif360preproc/',
                  '../data/npy_data_from_aif360/bank-aif360preproc/',
                  '../data/npy_data_from_aif360/default-aif360preproc/',
                  '../data/npy_data_from_aif360/heart-aif360preproc/',
                  '../data/npy_data_from_aif360/student-aif360preproc/',
                  '../data/npy_data_from_aif360/meps-aif360preproc/',
                  '../data/npy_data_from_aif360/home_credit-aif360preproc/'
                  ]

if __name__ == '__main__':
    df = pd.read_csv('../../data/raw/home/home.csv')
    # map_unique_as_num('NAME_TYPE_SUITE', df)
    # m = df['AMT_ANNUITY'].mean
    df = df.dropna()
    clip_numeric_normalize(df, 'AMT_ANNUITY')
    # binary_numeric_by_mean(df, 'AMT_ANNUITY')
    print('done')
