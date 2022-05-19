
import sys
sys.path.append("../")
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_compas, load_preproc_data_german, convert_two_dims_labels

from aif360.algorithms.inprocessing.adversarial_debiasing_dnn5 import AdversarialDebiasingDnn5
if __name__ == '__main__':
    # adult protected attr： sex race
    dataset_orig = load_preproc_data_adult(convert=True)

    privileged_groups = [{'sex': 1}] # 在具体计算指标的时候调用，可以替换成race
    unprivileged_groups = [{'sex': 0}]
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.8], shuffle=True)
    # 转换成二维的labels用以适应DNN的训练和测试，但是在其他AIF360应用的地方，是一维的
    train_converted_labels = convert_two_dims_labels()