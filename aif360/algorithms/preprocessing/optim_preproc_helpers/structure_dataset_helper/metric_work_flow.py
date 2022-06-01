import sys

from aif360.datasets.binary_label_dataset import BinaryLabelDataset
from aif360.metrics.binary_label_dataset_metric import BinaryLabelDatasetMetric
from aif360.metrics.classification_metric import ClassificationMetric

sys.path.append("../")
import numpy as np
import pandas as pd
from . import npy_to_df as ntd
from . import attr_map_helper as amh

data_set_list = ['adult', 'compas', 'german', 'bank',
                 'default', 'heart', 'student',
                 'meps15', 'meps16']
data_set_list_compat = ['adult', 'adult',
                        'compas', 'compas',
                        'german', 'bank',
                        'default', 'heart', 'student',
                        'meps15', 'meps16']
dataset_with_d_attr_list = ['adult_race', 'adult_sex',
                            'compas_race', 'compas_sex',
                            'german_sex',
                            'bank_age',
                            'default_sex',
                            'heart_age',
                            'student_sex',
                            'meps15_race',
                            'meps16_race']
def dataset_list():
    return data_set_list

def dataset_list_compat():
    return data_set_list_compat

def dataset_d_list():
    return dataset_with_d_attr_list

def construct_d_maps(dataset_with_d_name='adult_race'):
    # TODO 下面这个拿到的是纯str， 不是map ，还需写一个用到meta 的map
    all_protected_attribute = amh.get_d_raw_maps()[dataset_with_d_name]
    all_protected_attribute_maps = {all_protected_attribute: amh.get_d_meta_maps()[dataset_with_d_name]}
    y, d, p, up = amh.get_all_map_item_from_txt()
    D_features = (d[dataset_with_d_name])
    all_privileged_classes = {d[dataset_with_d_name]: (p[dataset_with_d_name])}
    all_unprivileged_classes = {d[dataset_with_d_name]: (up[dataset_with_d_name])}
    return all_protected_attribute, all_protected_attribute_maps, all_privileged_classes, \
           D_features, all_unprivileged_classes


def construct_p_up_group(dataset_with_d_name='adult_race'):
    y, d, p, up = amh.get_all_map_item_from_txt()
    all_privileged_classes = {d[dataset_with_d_name]: (p[dataset_with_d_name])}
    all_unprivileged_classes = {d[dataset_with_d_name]: (up[dataset_with_d_name])}
    p = [all_privileged_classes]
    u = [all_unprivileged_classes]
    return p, u


def structring_classification_dataset_from_npy(x_path, y_path, pre_path,
                                               dataset_name='adult', dataset_with_d_name='adult_race', print_bool=True):
    x_arr = np.load(x_path)
    y_arr = np.load(y_path)
    pre_arr = np.load(pre_path)
    df = ntd.get_df_from_npy_x_y(x_arr, y_arr, dataset_name=dataset_name)
    df_pre = ntd.get_df_from_npy_x_y(x_arr, pre_arr, dataset_name=dataset_name)
    attr_names = amh.get_all_attr_list()[data_set_list.index(dataset_name)]  # df.cols也可以
    # feature_names = df.columns
    all_protected_attribute, all_protected_attribute_maps, all_privileged_classes, \
    D_features, all_unprivileged_classes = construct_d_maps(dataset_with_d_name)

    def convert(classes, x):
        if len(list(classes[x])) == 1:
            classes[x] = int(classes[x])
        else:
            classes[x] = (classes[x])
        return classes[x]
    if dataset_with_d_name in ['bank_age']:
        unp_pro_attrs = [[1.0, 2.0]]
        p_pro_attrs = [[3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]]
        meta_pro_attr_maps = [all_protected_attribute_maps[x] for x in [D_features]]
    elif dataset_with_d_name in ['heart_age']:
        unp_pro_attrs = [[3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]]
        p_pro_attrs = [[1.0, 2.0]]
        meta_pro_attr_maps = [all_protected_attribute_maps[x] for x in [D_features]]
    else:
        unp_pro_attrs = [[convert(all_unprivileged_classes, x) for x in [D_features]]]
        p_pro_attrs = [[convert(all_privileged_classes, x) for x in [D_features]]]
        meta_pro_attr_maps = [all_protected_attribute_maps[x] for x in [D_features]]
    orig_dataset = BinaryLabelDataset(favorable_label=1,
                                      unfavorable_label=0,
                                      df=df,
                                      label_names=[attr_names[-1]],
                                      protected_attribute_names=[D_features],
                                      instance_weights_name=None,
                                      scores_names=[],  # 未来需要将predict用作score
                                      unprivileged_protected_attributes=unp_pro_attrs,
                                      privileged_protected_attributes=p_pro_attrs,
                                      metadata={'label_maps': [{1: 'favorable', 0: 'unfavorable'}],
                                                'protected_attribute_maps': meta_pro_attr_maps}
                                      )
    predict_dataset = BinaryLabelDataset(favorable_label=1,
                                         unfavorable_label=0,
                                         df=df_pre,
                                         label_names=[attr_names[-1]],
                                         protected_attribute_names=[D_features],
                                         instance_weights_name=None,
                                         scores_names=[],  # 未来需要将predict用作score
                                         unprivileged_protected_attributes=unp_pro_attrs,
                                         privileged_protected_attributes=p_pro_attrs,
                                         metadata={'label_maps': [{1: 'favorable', 0: 'unfavorable'}],
                                                   'protected_attribute_maps': meta_pro_attr_maps}
                                         )
    p, u = construct_p_up_group(dataset_with_d_name=dataset_with_d_name)
    dm = ClassificationMetric(orig_dataset, predict_dataset, unprivileged_groups=u, privileged_groups=p)
    if print_bool:
        print(
            'Test set: SPD in outcomes between unprivileged and privileged groups = %f' % dm.statistical_parity_difference())
        print('Test set: DI in outcomes between unprivileged and privileged groups = %f' % dm.disparate_impact())
        print('Test set: equal opportunity in outcomes between unprivileged and privileged groups = %f'
              % dm.equal_opportunity_difference())
        print('Test set: average odds diff in outcomes between unprivileged and privileged groups = %f'
              % dm.average_odds_difference())
        print('Test set: average absolute odds diff in outcomes between unprivileged and privileged groups = %f'
              % dm.average_abs_odds_difference())
        print('Test set: accuracy in outcomes between unprivileged and privileged groups = %f'
              % dm.accuracy())

        print(
            'Test set: pos_nums in outcomes between unprivileged and privileged groups = %f' % dm.num_positives())
        print(
            'Test set: pre_pos_nums in outcomes between unprivileged and privileged groups = %f' % dm.num_pred_positives())
        print('Test set: neg_nums in outcomes between unprivileged and privileged groups = %f'
              % dm.num_negatives())
        print('Test set: average odds diff in outcomes between unprivileged and privileged groups = %f'
              % dm.num_pred_negatives())

    return dm

def structring_classification_dataset_from_npy_array(x_, y_, pre_,
                                               dataset_name='adult', dataset_with_d_name='adult_race', print_bool=True):
    x_arr = x_
    y_arr = y_
    pre_arr = pre_
    df = ntd.get_df_from_npy_x_y(x_arr, y_arr, dataset_name=dataset_name)
    df_pre = ntd.get_df_from_npy_x_y(x_arr, pre_arr, dataset_name=dataset_name)
    attr_names = amh.get_all_attr_list()[data_set_list.index(dataset_name)]  # df.cols也可以
    # feature_names = df.columns
    all_protected_attribute, all_protected_attribute_maps, all_privileged_classes, \
    D_features, all_unprivileged_classes = construct_d_maps(dataset_with_d_name)

    def convert(classes, x):
        if len(list(classes[x])) == 1:
            classes[x] = int(classes[x])
        else:
            classes[x] = (classes[x])
        return classes[x]
    if dataset_with_d_name in ['bank_age']:
        unp_pro_attrs = [[1.0, 2.0]]
        p_pro_attrs = [[3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]]
        meta_pro_attr_maps = [all_protected_attribute_maps[x] for x in [D_features]]
    elif dataset_with_d_name in ['heart_age']:
        unp_pro_attrs = [[3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]]
        p_pro_attrs = [[1.0, 2.0]]
        meta_pro_attr_maps = [all_protected_attribute_maps[x] for x in [D_features]]
    else:
        unp_pro_attrs = [[convert(all_unprivileged_classes, x) for x in [D_features]]]
        p_pro_attrs = [[convert(all_privileged_classes, x) for x in [D_features]]]
        meta_pro_attr_maps = [all_protected_attribute_maps[x] for x in [D_features]]
    orig_dataset = BinaryLabelDataset(favorable_label=1,
                                      unfavorable_label=0,
                                      df=df,
                                      label_names=[attr_names[-1]],
                                      protected_attribute_names=[D_features],
                                      instance_weights_name=None,
                                      scores_names=[],  # 未来需要将predict用作score
                                      unprivileged_protected_attributes=unp_pro_attrs,
                                      privileged_protected_attributes=p_pro_attrs,
                                      metadata={'label_maps': [{1: 'favorable', 0: 'unfavorable'}],
                                                'protected_attribute_maps': meta_pro_attr_maps}
                                      )
    predict_dataset = BinaryLabelDataset(favorable_label=1,
                                         unfavorable_label=0,
                                         df=df_pre,
                                         label_names=[attr_names[-1]],
                                         protected_attribute_names=[D_features],
                                         instance_weights_name=None,
                                         scores_names=[],  # 未来需要将predict用作score
                                         unprivileged_protected_attributes=unp_pro_attrs,
                                         privileged_protected_attributes=p_pro_attrs,
                                         metadata={'label_maps': [{1: 'favorable', 0: 'unfavorable'}],
                                                   'protected_attribute_maps': meta_pro_attr_maps}
                                         )
    p, u = construct_p_up_group(dataset_with_d_name=dataset_with_d_name)
    dm = ClassificationMetric(orig_dataset, predict_dataset, unprivileged_groups=u, privileged_groups=p)
    if print_bool:
        print(
            'Test set: SPD in outcomes between unprivileged and privileged groups = %f' % dm.statistical_parity_difference())
        print('Test set: DI in outcomes between unprivileged and privileged groups = %f' % dm.disparate_impact())
        print('Test set: equal opportunity in outcomes between unprivileged and privileged groups = %f'
              % dm.equal_opportunity_difference())
        print('Test set: average odds diff in outcomes between unprivileged and privileged groups = %f'
              % dm.average_odds_difference())
        print('Test set: average absolute odds diff in outcomes between unprivileged and privileged groups = %f'
              % dm.average_abs_odds_difference())
        print('Test set: accuracy in outcomes between unprivileged and privileged groups = %f'
              % dm.accuracy())

        print(
            'Test set: pos_nums in outcomes between unprivileged and privileged groups = %f' % dm.num_positives())
        print(
            'Test set: pre_pos_nums in outcomes between unprivileged and privileged groups = %f' % dm.num_pred_positives())
        print('Test set: neg_nums in outcomes between unprivileged and privileged groups = %f'
              % dm.num_negatives())
        print('Test set: average odds diff in outcomes between unprivileged and privileged groups = %f'
              % dm.num_pred_negatives())

    return dm


if __name__ == '__main__':
    for i in range(len(data_set_list_compat)):
        dataset_name = data_set_list_compat[i]
        dataset_name_d = dataset_with_d_attr_list[i]
        print('==============%s================' % dataset_name_d)
        x_path = '../../../../data/npy_data/' + dataset_name + '-aif360preproc/features-test.npy'
        y_path = '../../../../data/npy_data/' + dataset_name + '-aif360preproc/labels-test.npy'
        pre_path = '../../../../data/npy_data/' + dataset_name + '-aif360preproc/labels-test.npy'
        dm = structring_classification_dataset_from_npy(x_path, y_path, pre_path,
                                                        dataset_name=dataset_name,
                                                        dataset_with_d_name=dataset_name_d,
                                                        print_bool=True)

    print('done')
