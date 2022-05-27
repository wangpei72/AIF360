import sys

sys.path.append("../")
import numpy as np
# 'home', 还没做好
data_set_list = ['adult', 'compas', 'german', 'bank',
                 'default', 'heart', 'student',
                  'meps15', 'meps16']

def get_attr_from_txt(dataset_name):
    l_s = ''
    with open('attr_files/' + dataset_name + '_attr.txt', 'r+') as f:
        lines = f.read()
        lines = lines.replace("\n", "")
        l_s = lines.split(',')
        f.close()
    return l_s

def get_all_attr_list(data_set_list = ['adult', 'compas', 'german', 'bank',
                 'default', 'heart', 'student',
                  'meps15', 'meps16']):
    all_ls = []
    for i in data_set_list:
        all_ls.append(get_attr_from_txt(i))
    # print(all_ls)
    return all_ls

if __name__ == '__main__':
    all_ls = get_all_attr_list()