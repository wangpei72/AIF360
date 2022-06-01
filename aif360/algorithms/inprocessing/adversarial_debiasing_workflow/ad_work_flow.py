import sys

sys.path.append("../")
import numpy as np
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import *
import tensorflow.compat.v1 as tf
from aif360.algorithms.inprocessing.adversarial_debiasing_dnn5 import AdversarialDebiasingDnn5
from dnn5_predicttest import print_metric, print_test_set_classifier_metric

# 1 先指定所有的11类数据集
# 2 使用map等手段定义好语义信息，不对，这是在load函数里定义到类的
# 2 重新整理load函数得到的类，传入所有的语义信息，并且编写一个类似于switch的总函数，传入str调用对应的load函数，打印初始公平性metric
# 3 整理ad类必要的流程，先是fit，打印对应的公平性metric
# 4 ad类接下来的流程是predict， 获得了predict之后的数据集类，进行classifier类的指标打印
# 5 整理获得的数据集，在aif360中进行nc数据的计算，所以需要将lsa dsa引入至aif360中

dataset_with_d_list = dataset_with_d_name_list()
pri_cond_map = {
             'adult_race': [{'race': 1}],
             'adult_sex': [{'sex': 1}],
             'compas_race': [{'race': 1}],
             'compas_sex': [{'sex': 1}],
             'german_sex': [{'personal_status': 1}],
             'bank_age': [{'age': [3,4,5,6,7,8,9]}],
             'default_sex': [{'X2': 1}],
             'heart_age': [ {'age': [2,3]}],
             'student_sex': [{'sex': 1}],
             'meps15_race': [{'RACE': 1}],
             'meps16_race': [{'RACE': 1}]}
unpri_cond_map = {
             'adult_race': [{'race': 0}],
             'adult_sex': [{'sex': 0}],
             'compas_race': [{'race': 0}],
             'compas_sex': [{'sex': 0}],
             'german_sex': [{'personal_status': 0}],
             'bank_age': [{'age': [0,1,2,3]}],
             'default_sex': [{'X2': 0}],
             'heart_age': [ {'age': [4,5,6,7]}],
             'student_sex': [{'sex': 0}],
             'meps15_race': [{'RACE': 0}],
             'meps16_race': [{'RACE': 0}]
}

def train_adversary_debiasing_mods_dnn5(dataset_orig,
                                        dataset_d_name,
                                        privileged_groups = [{'sex': 1}],
                                        unprivileged_groups = [{'sex': 0}]):
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.8], shuffle=False)
    sess = tf.Session()
    print('train plain model starts...')
    plain_model = AdversarialDebiasingDnn5(
        privileged_groups=privileged_groups,
        unprivileged_groups=unprivileged_groups,
        scope_name='plain_classifier',
        num_epochs=1000,
        debias=False,
        dataset_d_name=dataset_d_name,
        sess=sess)
    plain_model.fit(dataset_orig_train)
    sess.close()
    tf.reset_default_graph()
    sess = tf.Session()
    print('train with adversary starts...')
    debiased_model = AdversarialDebiasingDnn5(privileged_groups=privileged_groups,
                                              unprivileged_groups=unprivileged_groups,
                                              scope_name='debiased_classifier',
                                              num_epochs=1000,
                                              debias=True,
                                              dataset_d_name=dataset_d_name,
                                              sess=sess)
    debiased_model.fit(dataset_orig_train)


def main():
    for i in range(len(dataset_with_d_list)):
        print('==============current dataset is %s ==================' % dataset_with_d_list[i])
        dataset_orig = load_preproc_data(dataset_with_d_list[i])
        train_adversary_debiasing_mods_dnn5(dataset_orig,
                                            dataset_d_name=dataset_with_d_list[i],
                                            privileged_groups=pri_cond_map[dataset_with_d_list[i]],
                                            unprivileged_groups=unpri_cond_map[dataset_with_d_list[i]])
    print('done with all ')


if __name__ == '__main__':
    main()



