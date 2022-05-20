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
    load_preproc_data_student, load_preproc_data_meps21, load_preproc_data_meps19, load_preproc_data_meps20

from aif360.algorithms.inprocessing.adversarial_debiasing_dnn5 import AdversarialDebiasingDnn5
import numpy as np

if __name__ == '__main__':
# adult protected attr： sex race
#     dataset_orig_adult = load_preproc_data_adult(convert=True)

    privileged_groups = [{'sex': 1}]  # 在具体计算指标的时候调用，可以替换成race age等
    unprivileged_groups = [{'sex': 0}]
    # privileged_groups = [{'race': 1}]
    # unprivileged_groups = [{'race': 0}]
    # privileged_groups = [{'age': 1}]
    # unprivileged_groups = [{'age': 0}]
#     dataset_orig_train, dataset_orig_test = dataset_orig_adult.split([0.8], shuffle=True)
#     # 转换成二维的labels用以适应DNN的训练和测试，但是在其他AIF360应用的地方，是一维的
#     train_converted_labels = convert_two_dims_labels(dataset_orig_train)
#     test_converted_labels = convert_two_dims_labels(dataset_orig_test)
#
#     np.save('../aif360/data/npy_data/adult-aif360preproc/2d-labels-train.npy', train_converted_labels)
#     np.save('../aif360/data/npy_data/adult-aif360preproc/2d-labels-test.npy', test_converted_labels)
#
# #     compas protected-attr : sex race
#     dataset_orig_compas = load_preproc_data_compas()
#
#     dataset_orig_train, dataset_orig_test = dataset_orig_compas.split([0.8], shuffle=True)
#     train_converted_labels = convert_two_dims_labels(dataset_orig_train)
#     test_converted_labels = convert_two_dims_labels(dataset_orig_test)
#
#     np.save('../aif360/data/npy_data/compas-aif360preproc/2d-labels-train.npy', train_converted_labels)
#     np.save('../aif360/data/npy_data/compas-aif360preproc/2d-labels-test.npy', test_converted_labels)
#
# # German credit protected attr : sex
#     dataset_orig_german = load_preproc_data_german()
#
#     dataset_orig_train, dataset_orig_test = dataset_orig_german.split([0.8], shuffle=True)
#     train_converted_labels = convert_two_dims_labels(dataset_orig_train)
#     test_converted_labels = convert_two_dims_labels(dataset_orig_test)
#
#     np.save('../aif360/data/npy_data/german-aif360preproc/2d-labels-train.npy', train_converted_labels)
#     np.save('../aif360/data/npy_data/german-aif360preproc/2d-labels-test.npy', test_converted_labels)
#


#     bank marketing attr: age
#     dataset_orig_bank = load_preproc_data_bank()
#     bank_orig_train, bank_orig_test = dataset_orig_bank.split([0.8], shuffle=True)
#     train_converted_labels = convert_two_dims_labels(bank_orig_train)
#     test_converted_labels = convert_two_dims_labels(bank_orig_test)
#     save_dir = '../aif360/data/npy_data/bank-aif360preproc/'
#     # if not os._exists(save_dir):
#     #     os.makedirs(save_dir)
#     np.save('../aif360/data/npy_data/bank-aif360preproc/labels-train.npy', bank_orig_train.labels)
#     np.save('../aif360/data/npy_data/bank-aif360preproc/labels-test.npy', bank_orig_test.labels)
#     np.save('../aif360/data/npy_data/bank-aif360preproc/features-train.npy', bank_orig_train.features)
#     np.save('../aif360/data/npy_data/bank-aif360preproc/features-test.npy', bank_orig_test.features)
#     np.save('../aif360/data/npy_data/bank-aif360preproc/2d-labels-train.npy', train_converted_labels)
#     np.save('../aif360/data/npy_data/bank-aif360preproc/2d-labels-test.npy', test_converted_labels)
#     print('done')

#     default credit card  pro-attr: age (<25/ >= 25)
#     dataset_orig_default = load_preproc_data_default()
#     default_orig_train, default_orig_test = dataset_orig_default.split([0.8], shuffle=False)
#     train_converted_labels = convert_two_dims_labels(default_orig_train)
#     test_converted_labels = convert_two_dims_labels(default_orig_test)
#     save_dir = '../aif360/data/npy_data/default-aif360preproc/'
#     if not os._exists(save_dir):
#         os.makedirs(save_dir)
#     np.save('../aif360/data/npy_data/default-aif360preproc/labels-train.npy', default_orig_train.labels)
#     np.save('../aif360/data/npy_data/default-aif360preproc/labels-test.npy', default_orig_test.labels)
#     np.save('../aif360/data/npy_data/default-aif360preproc/features-train.npy', default_orig_train.features)
#     np.save('../aif360/data/npy_data/default-aif360preproc/features-test.npy', default_orig_test.features)
#     np.save('../aif360/data/npy_data/default-aif360preproc/2d-labels-train.npy', train_converted_labels)
#     np.save('../aif360/data/npy_data/default-aif360preproc/2d-labels-test.npy', test_converted_labels)


# #    heart pro-attr: age (<54.4/ >= 54.4)
#     dataset_orig_heart = load_preproc_data_heart()
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

#    student pro-attr: sex (male is privileged)
#     dataset_orig_student = load_preproc_data_student()
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


#    meps19 pro-attr: race (white is privileged)
#     dataset_orig_meps19 = load_preproc_data_meps19()
#     meps19_orig_train, meps19_orig_test = dataset_orig_meps19.split([0.8], shuffle=False)
#     train_converted_labels = convert_two_dims_labels(meps19_orig_train)
#     test_converted_labels = convert_two_dims_labels(meps19_orig_test)
#     save_dir = '../aif360/data/npy_data/meps19-aif360preproc/'
#     if not os._exists(save_dir):
#         os.makedirs(save_dir)
#     np.save('../aif360/data/npy_data/meps19-aif360preproc/labels-train.npy', meps19_orig_train.labels)
#     np.save('../aif360/data/npy_data/meps19-aif360preproc/labels-test.npy', meps19_orig_test.labels)
#     np.save('../aif360/data/npy_data/meps19-aif360preproc/features-train.npy', meps19_orig_train.features)
#     np.save('../aif360/data/npy_data/meps19-aif360preproc/features-test.npy', meps19_orig_test.features)
#     np.save('../aif360/data/npy_data/meps19-aif360preproc/2d-labels-train.npy', train_converted_labels)
#     np.save('../aif360/data/npy_data/meps19-aif360preproc/2d-labels-test.npy', test_converted_labels)

#    meps20 pro-attr: race (white is privileged)
#     dataset_orig_meps20 = load_preproc_data_meps20()
#     meps20_orig_train, meps20_orig_test = dataset_orig_meps20.split([0.8], shuffle=False)
#     train_converted_labels = convert_two_dims_labels(meps20_orig_train)
#     test_converted_labels = convert_two_dims_labels(meps20_orig_test)
#     save_dir = '../aif360/data/npy_data/meps20-aif360preproc/'
#     if not os._exists(save_dir):
#         os.makedirs(save_dir)
#     np.save('../aif360/data/npy_data/meps20-aif360preproc/labels-train.npy', meps20_orig_train.labels)
#     np.save('../aif360/data/npy_data/meps20-aif360preproc/labels-test.npy', meps20_orig_test.labels)
#     np.save('../aif360/data/npy_data/meps20-aif360preproc/features-train.npy', meps20_orig_train.features)
#     np.save('../aif360/data/npy_data/meps20-aif360preproc/features-test.npy', meps20_orig_test.features)
#     np.save('../aif360/data/npy_data/meps20-aif360preproc/2d-labels-train.npy', train_converted_labels)
#     np.save('../aif360/data/npy_data/meps20-aif360preproc/2d-labels-test.npy', test_converted_labels)

    # meps 21 pro-attr: race (white is privileged)
    dataset_orig_meps21 = load_preproc_data_meps21()
    meps21_orig_train, meps21_orig_test = dataset_orig_meps21.split([0.8], shuffle=False)
    train_converted_labels = convert_two_dims_labels(meps21_orig_train)
    test_converted_labels = convert_two_dims_labels(meps21_orig_test)
    save_dir = '../aif360/data/npy_data/meps21-aif360preproc/'
    if not os._exists(save_dir):
        os.makedirs(save_dir)
    np.save('../aif360/data/npy_data/meps21-aif360preproc/labels-train.npy', meps21_orig_train.labels)
    np.save('../aif360/data/npy_data/meps21-aif360preproc/labels-test.npy', meps21_orig_test.labels)
    np.save('../aif360/data/npy_data/meps21-aif360preproc/features-train.npy', meps21_orig_train.features)
    np.save('../aif360/data/npy_data/meps21-aif360preproc/features-test.npy', meps21_orig_test.features)
    np.save('../aif360/data/npy_data/meps21-aif360preproc/2d-labels-train.npy', train_converted_labels)
    np.save('../aif360/data/npy_data/meps21-aif360preproc/2d-labels-test.npy', test_converted_labels)
    print('done')

