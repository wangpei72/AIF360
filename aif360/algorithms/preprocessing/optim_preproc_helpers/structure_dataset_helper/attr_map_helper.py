import os.path
import sys

sys.path.append("../")
import numpy as np

# 'home', 还没做好
data_set_list = ['adult', 'compas', 'german', 'bank',
                 'default', 'heart', 'student',
                 'meps15', 'meps16']

map_dataset_p_attr = {
    'adult': ['race', 'sex'],
    'compas': ['race', 'sex'],
    'german': ['sex'],
    'bank': ['age'],
    'default': ['sex'],
    'heart': ['age'],
    'student': ['sex'],
    'meps15': ['race'],
    'meps16': ['race']
}
d_map_categoric_attr = {'adult_race': 'race', 'adult_sex': 'sex', 'compas_race': 'race',
                       'compas_sex': 'sex', 'german_sex': 'sex', 'bank_age': 'age',
                       'default_sex': 'x2', 'heart_age': 'age', 'student_sex': 'sex',
                       'meps15_race': 'race', 'meps16_race': 'race'}

d_map_categoric_meta = {'adult_race': {1: 'White', 0: 'Non-white'},
                        'adult_sex': {1: 'Male', 0: 'Female'},
                        'compas_race': {1: 'Caucasian', 0: 'Not Caucasian'},
                        'compas_sex': {0: 'Male', 1: 'Female'},
                        'german_sex': {1: 'Male', 0: 'Female'},
                        'bank_age': {1: 'Old', 0: 'Young'},
                        'default_sex': {1: 'Male', 0: 'Female'},
                        'heart_age': {1: 'Young', 0: 'Old'},
                        'student_sex': {1: 'Male', 0: 'Female'},
                        'meps15_race': {1: 'White', 0: 'Non-white'},
                        'meps16_race': {1: 'White', 0: 'Non-white'}}

d_attr_meta_map = {
    'sex_compas': {0: 'Male', 1: 'Female'},
    'race_compas': {1: 'Caucasian', 0: 'Not Caucasian'},
    'sex': {1: 'Male', 0: 'Female'},
    "race": {1: 'White', 0: 'Non-white'},
    'age_heart': {1: 'Young', 0: 'Old'},
    'age': {1: 'Old', 0: 'Young'},
}

def get_d_meta_maps():
    return d_map_categoric_meta

def get_d_maps():
    return map_dataset_p_attr


def get_d_raw_maps():
    return d_map_categoric_attr


def get_attr_from_txt(dataset_name):
    l_s = ''
    # # train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    #                               '..', 'data', 'raw', 'adult', 'adult.data')
    attr_file_path  = os.path.join(os.path.dirname(os.path.abspath(__file__)),'attr_files', dataset_name+'_attr.txt')
    with open(attr_file_path, 'r+') as f:
        lines = f.read()
        lines = lines.lower()
        lines = lines.replace("\n", "")
        l_s = lines.split(',')
        f.close()
    return l_s


def get_all_attr_list(data_set_list=['adult', 'compas', 'german', 'bank',
                                     'default', 'heart', 'student',
                                     'meps15', 'meps16']):
    all_ls = []
    for i in data_set_list:
        all_ls.append(get_attr_from_txt(i))
    # print(all_ls)
    return all_ls


def get_map_from_txt_1st(dataset_name, index=0):
    l_s = ''
    attr_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'map_files', 'map_' + dataset_name +'_'+
                                  map_dataset_p_attr[dataset_name][index] + '.txt')
    with open(attr_file_path, 'r+') as f:
        lines = f.read()
        lines = lines.replace('\n', '')
        l_s = lines.split(',')
        f.close()
    y_map_item = l_s[1]
    d_map_item = l_s[3]
    p_map_item = l_s[5]
    up_map_item = l_s[7]
    return y_map_item, d_map_item, p_map_item, up_map_item


def get_all_map_item_from_txt(data_set_list=['adult', 'compas', 'german', 'bank',
                                             'default', 'heart', 'student',
                                             'meps15', 'meps16']):
    y_map_item_dict = {}
    d_map_item_dict = {}
    p_map_item_dict = {}
    up_map_item_dict = {}
    for i in data_set_list:
        y, d, p, up = get_map_from_txt_1st(i)
        y_map_item_dict[i + '_' + map_dataset_p_attr[i][0]] = (y)
        d_map_item_dict[i + '_' + map_dataset_p_attr[i][0]] = d
        p_map_item_dict[i + '_' + map_dataset_p_attr[i][0]] = (p)
        up_map_item_dict[i + '_' + map_dataset_p_attr[i][0]] = (up)
        if len(map_dataset_p_attr[i]) == 2:
            y_, d_, p_, up_ = get_map_from_txt_1st(i, 1)
            y_map_item_dict[i + '_' + map_dataset_p_attr[i][1]] = (y_)
            d_map_item_dict[i + '_' + map_dataset_p_attr[i][1]] = (d_)
            p_map_item_dict[i + '_' + map_dataset_p_attr[i][1]] = (p_)
            up_map_item_dict[i + '_' + map_dataset_p_attr[i][1]] = (up_)
    return y_map_item_dict, d_map_item_dict, p_map_item_dict, up_map_item_dict


if __name__ == '__main__':
    all_ls = get_all_attr_list()
    all_ymap, all_dmap, all_pmap, all_upmap = get_all_map_item_from_txt()
    print(all_dmap)
    print('done')
