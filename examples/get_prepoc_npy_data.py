import sys
sys.path.append("../")
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_compas, load_preproc_data_german, convert_two_dims_labels

from aif360.algorithms.inprocessing.adversarial_debiasing_dnn5 import AdversarialDebiasingDnn5
import numpy as np

if __name__ == '__main__':
# adult protected attr： sex race
    dataset_orig_adult = load_preproc_data_adult(convert=True)

    privileged_groups = [{'sex': 1}]  # 在具体计算指标的时候调用，可以替换成race age等
    unprivileged_groups = [{'sex': 0}]
    # privileged_groups = [{'race': 1}]
    # unprivileged_groups = [{'race': 0}]
    # privileged_groups = [{'age': 1}]
    # unprivileged_groups = [{'age': 0}]
    dataset_orig_train, dataset_orig_test = dataset_orig_adult.split([0.8], shuffle=True)
    # 转换成二维的labels用以适应DNN的训练和测试，但是在其他AIF360应用的地方，是一维的
    train_converted_labels = convert_two_dims_labels(dataset_orig_train)
    test_converted_labels = convert_two_dims_labels(dataset_orig_test)

    np.save('../aif360/data/npy_data/adult-aif360preproc/2d-labels-train.npy', train_converted_labels)
    np.save('../aif360/data/npy_data/adult-aif360preproc/2d-labels-test.npy', test_converted_labels)

#     compas protected-attr : sex race
    dataset_orig_compas = load_preproc_data_compas()

    dataset_orig_train, dataset_orig_test = dataset_orig_compas.split([0.8], shuffle=True)
    train_converted_labels = convert_two_dims_labels(dataset_orig_train)
    test_converted_labels = convert_two_dims_labels(dataset_orig_test)

    np.save('../aif360/data/npy_data/compas-aif360preproc/2d-labels-train.npy', train_converted_labels)
    np.save('../aif360/data/npy_data/compas-aif360preproc/2d-labels-test.npy', test_converted_labels)

# German credit protected attr : sex
    dataset_orig_german = load_preproc_data_german()

    dataset_orig_train, dataset_orig_test = dataset_orig_german.split([0.8], shuffle=True)
    train_converted_labels = convert_two_dims_labels(dataset_orig_train)
    test_converted_labels = convert_two_dims_labels(dataset_orig_test)

    np.save('../aif360/data/npy_data/german-aif360preproc/2d-labels-train.npy', train_converted_labels)
    np.save('../aif360/data/npy_data/german-aif360preproc/2d-labels-test.npy', test_converted_labels)
    print('done')
