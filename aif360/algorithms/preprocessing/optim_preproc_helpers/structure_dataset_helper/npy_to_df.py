import sys

sys.path.append("../")
import numpy as np
import pandas as pd
import get_attr_header as attr

data_set_list = ['adult', 'compas', 'german', 'bank',
                 'default', 'heart', 'student',
                  'meps15', 'meps16']


def get_df_from_npy_x_y(x, y, dataset_name='adult'):
    idx_in_set_list = data_set_list.index(dataset_name)
    cols = attr.get_all_attr_list()[idx_in_set_list]
    concanate = np.concatenate((x, y), axis=1)
    df = pd.DataFrame(concanate, columns=cols)
    return df


def get_x_y_pre_arr(dataset_name='adult'):
    test_x = np.load('../../../../data/npy_data/' + dataset_name + '-aif360preproc/features-test.npy')
    test_y = np.load('../../../../data/npy_data/' + dataset_name + '-aif360preproc/labels-test.npy')
    predict_y = np.load('../../../../data/npy_data/' + dataset_name + '-aif360preproc/labels-test.npy')
    return test_x, test_y, predict_y

def get_all_x_y_pre():
    x = []
    y = []
    pre = []
    for i in data_set_list:
        x.append(get_x_y_pre_arr(i)[0])
        y.append(get_x_y_pre_arr(i)[1])
        pre.append(get_x_y_pre_arr(i)[2])
    return x, y, pre

def get_all_df():
    all = get_all_x_y_pre()
    all_x = all[0]
    all_y = all[1]
    all_pre = all[2]
    all_df = []
    for i in range(len(all_x)):
        all_df.append(get_df_from_npy_x_y(all_x[i], all_y[i], dataset_name=data_set_list[i]))
    return all_df

if __name__ == '__main__':
    # all_attr_list = attr.get_all_attr_list()
    get_all_df()
    # adult
    # test_x = np.load('../../../../data/npy_data/adult-aif360preproc/features-test.npy')
    # test_y = np.load('../../../../data/npy_data/adult-aif360preproc/labels-test.npy')
    # 暂时用label来代替
    # predict_y = np.load('../../../../data/npy_data/adult-aif360preproc/labels-test.npy')

    # concanate = np.concatenate((test_x, test_y), axis=1)
    # df = pd.DataFrame(concanate, columns=all_attr_list[0])
    print('done')
