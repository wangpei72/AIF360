import os
import sys
sys.path.append("../")
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, \
    load_preproc_data_compas, load_preproc_data_german, \
    convert_two_dims_labels, load_preproc_data_bank, load_preproc_data_default, load_preproc_data_heart, \
    load_preproc_data_student,  load_preproc_data_meps15, load_preproc_data_meps16, \
    load_preproc_data_home_credit

from aif360.algorithms.inprocessing.adversarial_debiasing_dnn5 import AdversarialDebiasingDnn5
import numpy as np

if __name__ == '__main__':


    # privileged_groups = [{'sex': 1}]  # 在具体计算指标的时候调用，可以替换成race age等
    # unprivileged_groups = [{'sex': 0}]
    # privileged_groups = [{'race': 1}]
    # unprivileged_groups = [{'race': 0}]
    # privileged_groups = [{'age': 1}]
    # unprivileged_groups = [{'age': 0}]
    # adult protected attr： sex race
    # dataset_orig_adult = load_preproc_data_adult(protected_attributes=['race'])
    # adult_orig_train,  adult_orig_test = dataset_orig_adult.split([0.8], shuffle=False)
#     # 转换成二维的labels用以适应DNN的训练和测试，但是在其他AIF360应用的地方，是一维的
#     train_converted_labels = convert_two_dims_labels(adult_orig_train)
#     test_converted_labels = convert_two_dims_labels( adult_orig_test)
#     save_dir = '../aif360/data/npy_data/adult-aif360preproc/'
#     if not os._exists(save_dir):
#         os.makedirs(save_dir)
#     np.save('../aif360/data/npy_data/adult-aif360preproc/labels-train.npy', adult_orig_train.labels)
#     np.save('../aif360/data/npy_data/adult-aif360preproc/labels-test.npy', adult_orig_test.labels)
#     np.save('../aif360/data/npy_data/adult-aif360preproc/features-train.npy', adult_orig_train.features)
#     np.save('../aif360/data/npy_data/adult-aif360preproc/features-test.npy', adult_orig_test.features)
#     np.save('../aif360/data/npy_data/adult-aif360preproc/2d-labels-train.npy', train_converted_labels)
#     np.save('../aif360/data/npy_data/adult-aif360preproc/2d-labels-test.npy', test_converted_labels)
#
# #     compas protected-attr : sex race
#     dataset_orig_compas = load_preproc_data_compas(protected_attributes=['race'])
#     compas_orig_train, compas_orig_test = dataset_orig_compas.split([0.8], shuffle=False)
#     train_converted_labels = convert_two_dims_labels(compas_orig_train)
#     test_converted_labels = convert_two_dims_labels(compas_orig_test)
#     save_dir = '../aif360/data/npy_data/compas-aif360preproc/'
#     if not os._exists(save_dir):
#         os.makedirs(save_dir)
#     np.save('../aif360/data/npy_data/compas-aif360preproc/labels-train.npy', compas_orig_train.labels)
#     np.save('../aif360/data/npy_data/compas-aif360preproc/labels-test.npy', compas_orig_test.labels)
#     np.save('../aif360/data/npy_data/compas-aif360preproc/features-train.npy', compas_orig_train.features)
#     np.save('../aif360/data/npy_data/compas-aif360preproc/features-test.npy', compas_orig_test.features)
#     np.save('../aif360/data/npy_data/compas-aif360preproc/2d-labels-train.npy', train_converted_labels)
#     np.save('../aif360/data/npy_data/compas-aif360preproc/2d-labels-test.npy', test_converted_labels)
#     print('done')

# # German credit protected attr : sex (personal_status)
#     dataset_orig_german = load_preproc_data_german(protected_attributes=['personal_status'])
#     german_orig_train, german_orig_test = dataset_orig_german.split([0.8], shuffle=False)
#     train_converted_labels = convert_two_dims_labels(german_orig_train)
#     test_converted_labels = convert_two_dims_labels(german_orig_test)
#     save_dir = '../aif360/data/npy_data/german-aif360preproc/'
#     # if not os._exists(save_dir):
#     #     os.makedirs(save_dir)
#     np.save('../aif360/data/npy_data/german-aif360preproc/labels-train.npy', german_orig_train.labels)
#     np.save('../aif360/data/npy_data/german-aif360preproc/labels-test.npy', german_orig_test.labels)
#     np.save('../aif360/data/npy_data/german-aif360preproc/features-train.npy', german_orig_train.features)
#     np.save('../aif360/data/npy_data/german-aif360preproc/features-test.npy', german_orig_test.features)
#     np.save('../aif360/data/npy_data/german-aif360preproc/2d-labels-train.npy', train_converted_labels)
#     np.save('../aif360/data/npy_data/german-aif360preproc/2d-labels-test.npy', test_converted_labels)
#     print('done')


    # bank marketing attr: age(12/345678)
    # dataset_orig_bank = load_preproc_data_bank(protected_attributes=['age'])
    # bank_orig_train, bank_orig_test = dataset_orig_bank.split([0.8], shuffle=False)
    # train_converted_labels = convert_two_dims_labels(bank_orig_train)
    # test_converted_labels = convert_two_dims_labels(bank_orig_test)
    # save_dir = '../aif360/data/npy_data/bank-aif360preproc/'
    # # if not os._exists(save_dir):
    # #     os.makedirs(save_dir)
    # np.save('../aif360/data/npy_data/bank-aif360preproc/labels-train.npy', bank_orig_train.labels)
    # np.save('../aif360/data/npy_data/bank-aif360preproc/labels-test.npy', bank_orig_test.labels)
    # np.save('../aif360/data/npy_data/bank-aif360preproc/features-train.npy', bank_orig_train.features)
    # np.save('../aif360/data/npy_data/bank-aif360preproc/features-test.npy', bank_orig_test.features)
    # np.save('../aif360/data/npy_data/bank-aif360preproc/2d-labels-train.npy', train_converted_labels)
    # np.save('../aif360/data/npy_data/bank-aif360preproc/2d-labels-test.npy', test_converted_labels)
    # print('done')

#     default credit card  pro-attr:sex
#     dataset_orig_default = load_preproc_data_default(protected_attributes=['X2'])
#     default_orig_train, default_orig_test = dataset_orig_default.split([0.8], shuffle=False)
#     train_converted_labels = convert_two_dims_labels(default_orig_train)
#     test_converted_labels = convert_two_dims_labels(default_orig_test)
#     save_dir = '../aif360/data/npy_data/default-aif360preproc/'
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     np.save('../aif360/data/npy_data/default-aif360preproc/labels-train.npy', default_orig_train.labels)
#     np.save('../aif360/data/npy_data/default-aif360preproc/labels-test.npy', default_orig_test.labels)
#     np.save('../aif360/data/npy_data/default-aif360preproc/features-train.npy', default_orig_train.features)
#     np.save('../aif360/data/npy_data/default-aif360preproc/features-test.npy', default_orig_test.features)
#     np.save('../aif360/data/npy_data/default-aif360preproc/2d-labels-train.npy', train_converted_labels)
#     np.save('../aif360/data/npy_data/default-aif360preproc/2d-labels-test.npy', test_converted_labels)
#     print('done')

# #    heart pro-attr: age (23 / 4567)
#     dataset_orig_heart = load_preproc_data_heart(protected_attributes=['age'])
#     heart_orig_train, heart_orig_test = dataset_orig_heart.split([0.8], shuffle=False)
#     train_converted_labels = convert_two_dims_labels(heart_orig_train)
#     test_converted_labels = convert_two_dims_labels(heart_orig_test)
#     save_dir = '../aif360/data/npy_data/heart-aif360preproc/'
#     if not os._exists(save_dir):
#         os.makedirs(save_dir)
#     np.save('../aif360/data/npy_data/heart-aif360preproc/labels-train.npy', heart_orig_train.labels)
#     np.save('../aif360/data/npy_data/heart-aif360preproc/labels-test.npy', heart_orig_test.labels)
#     np.save('../aif360/data/npy_data/heart-aif360preproc/features-train.npy', heart_orig_train.features)
#     np.save('../aif360/data/npy_data/heart-aif360preproc/features-test.npy', heart_orig_test.features)
#     np.save('../aif360/data/npy_data/heart-aif360preproc/2d-labels-train.npy', train_converted_labels)
#     np.save('../aif360/data/npy_data/heart-aif360preproc/2d-labels-test.npy', test_converted_labels)
#     print('done')

#    student pro-attr: sex (male is privileged)
#     dataset_orig_student = load_preproc_data_student(protected_attributes=['sex'])
#     student_orig_train, student_orig_test = dataset_orig_student.split([0.8], shuffle=False)
#     train_converted_labels = convert_two_dims_labels(student_orig_train)
#     test_converted_labels = convert_two_dims_labels(student_orig_test)
#     save_dir = '../aif360/data/npy_data/student-aif360preproc/'
#     if not os._exists(save_dir):
#         os.makedirs(save_dir)
#     np.save('../aif360/data/npy_data/student-aif360preproc/labels-train.npy', student_orig_train.labels)
#     np.save('../aif360/data/npy_data/student-aif360preproc/labels-test.npy', student_orig_test.labels)
#     np.save('../aif360/data/npy_data/student-aif360preproc/features-train.npy', student_orig_train.features)
#     np.save('../aif360/data/npy_data/student-aif360preproc/features-test.npy', student_orig_test.features)
#     np.save('../aif360/data/npy_data/student-aif360preproc/2d-labels-train.npy', train_converted_labels)
#     np.save('../aif360/data/npy_data/student-aif360preproc/2d-labels-test.npy', test_converted_labels)
#     print('done')

#    meps15 pro-attr: race (white is privileged)
#     dataset_orig_meps15 = load_preproc_data_meps15(protected_attributes=['RACE'])
#     meps15_orig_train, meps15_orig_test = dataset_orig_meps15.split([0.8], shuffle=False)
#     train_converted_labels = convert_two_dims_labels(meps15_orig_train)
#     test_converted_labels = convert_two_dims_labels(meps15_orig_test)
#     save_dir = '../aif360/data/npy_data/meps15-aif360preproc/'
# 
#     if not os._exists(save_dir):
#         os.makedirs(save_dir)
#     np.save('../aif360/data/npy_data/meps15-aif360preproc/labels-train.npy', meps15_orig_train.labels)
#     np.save('../aif360/data/npy_data/meps15-aif360preproc/labels-test.npy', meps15_orig_test.labels)
#     np.save('../aif360/data/npy_data/meps15-aif360preproc/features-train.npy', meps15_orig_train.features)
#     np.save('../aif360/data/npy_data/meps15-aif360preproc/features-test.npy', meps15_orig_test.features)
#     np.save('../aif360/data/npy_data/meps15-aif360preproc/2d-labels-train.npy', train_converted_labels)
#     np.save('../aif360/data/npy_data/meps15-aif360preproc/2d-labels-test.npy', test_converted_labels)
#     print('done')

#    meps16 pro-attr: race (white is privileged)
    dataset_orig_meps16 = load_preproc_data_meps16(protected_attributes=['RACE'])
#     meps16_orig_train, meps16_orig_test = dataset_orig_meps16.split([0.8], shuffle=False)
#     train_converted_labels = convert_two_dims_labels(meps16_orig_train)
#     test_converted_labels = convert_two_dims_labels(meps16_orig_test)
#     save_dir = '../aif360/data/npy_data/meps16-aif360preproc/'
#     if not os._exists(save_dir):
#         os.makedirs(save_dir)
#     np.save('../aif360/data/npy_data/meps16-aif360preproc/labels-train.npy', meps16_orig_train.labels)
#     np.save('../aif360/data/npy_data/meps16-aif360preproc/labels-test.npy', meps16_orig_test.labels)
#     np.save('../aif360/data/npy_data/meps16-aif360preproc/features-train.npy', meps16_orig_train.features)
#     np.save('../aif360/data/npy_data/meps16-aif360preproc/features-test.npy', meps16_orig_test.features)
#     np.save('../aif360/data/npy_data/meps16-aif360preproc/2d-labels-train.npy', train_converted_labels)
#     np.save('../aif360/data/npy_data/meps16-aif360preproc/2d-labels-test.npy', test_converted_labels)
    print('done')

    # meps 21 pro-attr: race (white is privileged)
    # dataset_orig_meps21 = load_preproc_data_meps21()
    # meps21_orig_train, meps21_orig_test = dataset_orig_meps21.split([0.8], shuffle=False)
    # train_converted_labels = convert_two_dims_labels(meps21_orig_train)
    # test_converted_labels = convert_two_dims_labels(meps21_orig_test)
    # save_dir = '../aif360/data/npy_data/meps21-aif360preproc/'
    # if not os._exists(save_dir):
    #     os.makedirs(save_dir)
    # np.save('../aif360/data/npy_data/meps21-aif360preproc/labels-train.npy', meps21_orig_train.labels)
    # np.save('../aif360/data/npy_data/meps21-aif360preproc/labels-test.npy', meps21_orig_test.labels)
    # np.save('../aif360/data/npy_data/meps21-aif360preproc/features-train.npy', meps21_orig_train.features)
    # np.save('../aif360/data/npy_data/meps21-aif360preproc/features-test.npy', meps21_orig_test.features)
    # np.save('../aif360/data/npy_data/meps21-aif360preproc/2d-labels-train.npy', train_converted_labels)
    # np.save('../aif360/data/npy_data/meps21-aif360preproc/2d-labels-test.npy', test_converted_labels)


    # concat 19 20 21 to one full dataset
    # dataset_orig_meps15 = load_preproc_data_meps15()
    # meps15_orig_train, meps15_orig_test = dataset_orig_meps15.split([0.8], shuffle=False)
    # train_converted_labels19 = convert_two_dims_labels(meps15_orig_train)
    # test_converted_labels19 = convert_two_dims_labels(meps15_orig_test)
    # dataset_orig_meps16 = load_preproc_data_meps16()
    # meps16_orig_train, meps16_orig_test = dataset_orig_meps16.split([0.8], shuffle=False)
    # train_converted_labels20 = convert_two_dims_labels(meps16_orig_train)
    # test_converted_labels20 = convert_two_dims_labels(meps16_orig_test)
    #
    # dataset_orig_meps21 = load_preproc_data_meps21()
    # meps21_orig_train, meps21_orig_test = dataset_orig_meps21.split([0.8], shuffle=False)
    # train_converted_labels21 = convert_two_dims_labels(meps21_orig_train)
    # test_converted_labels21 = convert_two_dims_labels(meps21_orig_test)
    #
    # concat_train_feat = np.concatenate((meps15_orig_train.features, meps16_orig_train.features, meps21_orig_train.features),
    #                                   axis=0)
    # concat_test_feat = np.concatenate((meps15_orig_test.features, meps16_orig_test.features, meps21_orig_test.features),
    #                                     axis=0)
    # concat_train_labels = np.concatenate((meps15_orig_train.labels, meps16_orig_train.labels, meps21_orig_train.labels),
    #                                   axis=0)
    # concat_test_labels = np.concatenate((meps15_orig_test.labels, meps16_orig_test.labels, meps21_orig_test.labels),
    #                                   axis=0)
    # concat_2d_train_labels = np.concatenate((train_converted_labels19, train_converted_labels20, train_converted_labels21),
    #                                   axis=0)
    # concat_2d_test_labels = np.concatenate((test_converted_labels19, test_converted_labels20, test_converted_labels21),
    #                                   axis=0)
    #
    # save_dir = '../aif360/data/npy_data/meps-aif360preproc/'
    # if not os._exists(save_dir):
    #     os.makedirs(save_dir)
    # np.save('../aif360/data/npy_data/meps-aif360preproc/labels-train.npy', concat_train_labels)
    # np.save('../aif360/data/npy_data/meps-aif360preproc/labels-test.npy', concat_test_labels)
    # np.save('../aif360/data/npy_data/meps-aif360preproc/features-train.npy', concat_train_feat)
    # np.save('../aif360/data/npy_data/meps-aif360preproc/features-test.npy', concat_test_feat)
    # np.save('../aif360/data/npy_data/meps-aif360preproc/2d-labels-train.npy', concat_2d_train_labels)
    # np.save('../aif360/data/npy_data/meps-aif360preproc/2d-labels-test.npy', concat_2d_test_labels)


    # dataset_orig_home_credit = load_preproc_data_home_credit()
    # home_credit_orig_train, home_credit_orig_test = dataset_orig_home_credit.split([0.8], shuffle=False)
    # train_converted_labels = convert_two_dims_labels(home_credit_orig_train)
    # test_converted_labels = convert_two_dims_labels(home_credit_orig_test)
    # save_dir = '../aif360/data/npy_data/home_credit-aif360preproc/'
    # if not os._exists(save_dir):
    #     os.makedirs(save_dir)
    # np.save('../aif360/data/npy_data/home_credit-aif360preproc/labels-train.npy', home_credit_orig_train.labels)
    # np.save('../aif360/data/npy_data/home_credit-aif360preproc/labels-test.npy', home_credit_orig_test.labels)
    # np.save('../aif360/data/npy_data/home_credit-aif360preproc/features-train.npy', home_credit_orig_train.features)
    # np.save('../aif360/data/npy_data/home_credit-aif360preproc/features-test.npy', home_credit_orig_test.features)
    # np.save('../aif360/data/npy_data/home_credit-aif360preproc/2d-labels-train.npy', train_converted_labels)
    # np.save('../aif360/data/npy_data/home_credit-aif360preproc/2d-labels-test.npy', test_converted_labels)
    # print('done')

