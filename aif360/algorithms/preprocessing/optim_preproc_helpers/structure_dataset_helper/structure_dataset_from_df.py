import sys

sys.path.append("../")
import numpy as np
import npy_to_df as ntd
from aif360.datasets.structured_dataset import StructuredDataset
from aif360.datasets.binary_label_dataset import BinaryLabelDataset
from aif360.metrics.binary_label_dataset_metric import BinaryLabelDatasetMetric
data_set_list = ['adult', 'compas', 'german', 'bank',
                 'default', 'heart', 'student',
                  'meps15', 'meps16']

if __name__ == '__main__':
    all_df = ntd.get_all_df()
    for i in data_set_list:
        df_tmp = all_df[data_set_list.index(i)]
    # TODO 还需要对所有必要的map进行一个汇总 和获取的helper类 这样我们就可以start所有的循环
    all_protected_attribute_maps = {'sex': {1: 'Male', 0: 'Female'},
                                    "race": {1: 'White', 0: 'Non-white'}}
    all_privileged_classes = {'sex': [1],
                              "race": [1]}
    D_features = ['sex', 'race']
    # test_set_adult = StructuredDataset(
    #                 df=df_adult,
    #                 label_names=['income-per-year'],
    #                 protected_attribute_names=['sex', 'race'],
    #                 instance_weights_name=None,
    #                 scores_names=[],  # 未来需要将predict用作score
    #                 unprivileged_protected_attributes=[[0], [0]],
    #                 privileged_protected_attributes=[[1], [1]],
    #                 metadata={'label_maps': [{1: '>50K', 0: '<=50K'}],
    #               'protected_attribute_maps': [all_protected_attribute_maps[x]
    #                             for x in D_features]})
    #
    # bin_test_set = BinaryLabelDataset(favorable_label=1, unfavorable_label=0,
    #                                   df=df_adult,
    #                                   label_names=['income-per-year'],
    #                                   protected_attribute_names=['sex', 'race'],
    #                                   instance_weights_name=None,
    #                                   scores_names=[],  # 未来需要将predict用作score
    #                                   unprivileged_protected_attributes=[[0], [0]],
    #                                   privileged_protected_attributes=[[1], [1]],
    #                                   metadata={'label_maps': [{1: '>50K', 0: '<=50K'}],
    #                                             'protected_attribute_maps': [all_protected_attribute_maps[x]
    #                                                                          for x in D_features]}
    #                                   )
    # p = [{'sex': 1}]
    # u = [{'sex': 0}]
    # dm = BinaryLabelDatasetMetric(bin_test_set, unprivileged_groups=u,
    #                    privileged_groups=p)
    #
    # print('Test set: Difference in mean outcomes between unprivileged and privileged groups = %f' %
    #       dm.mean_difference())
    print('done')
