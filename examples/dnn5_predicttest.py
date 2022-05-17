
import sys
sys.path.append("../")
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_compas, load_preproc_data_german

from aif360.algorithms.inprocessing.adversarial_debiasing_dnn5 import AdversarialDebiasingDnn5

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score

from IPython.display import Markdown, display
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

print('hii')



# print out some labels, names, etc.
def print_metric(dataset_train, dataset_test, orig_test):
    # mutated train set, mutated test set, origin test set
    metric_dataset_nodebiasing_train = BinaryLabelDatasetMetric(dataset_train,
                                                                unprivileged_groups=unprivileged_groups,
                                                                privileged_groups=privileged_groups)

    print(
        "Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_train.mean_difference())

    metric_dataset_nodebiasing_test = BinaryLabelDatasetMetric(dataset_test,
                                                               unprivileged_groups=unprivileged_groups,
                                                               privileged_groups=privileged_groups)

    print(
        "Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_test.mean_difference())

    print("Model: classification metrics")
    classified_metric_test = ClassificationMetric(orig_test,
                                                dataset_test,
                                                unprivileged_groups=unprivileged_groups,
                                                privileged_groups=privileged_groups)
    print("Test set: Classification accuracy = %f" % classified_metric_test.accuracy())
    TPR = classified_metric_test.true_positive_rate()
    TNR = classified_metric_test.true_negative_rate()
    bal_acc_nodebiasing_test = 0.5 * (TPR + TNR)
    print("Test set: Balanced classification accuracy = %f" % bal_acc_nodebiasing_test)
    print("Test set: Disparate impact = %f" % classified_metric_test.disparate_impact())
    print(
        "Test set: Equal opportunity difference = %f" % classified_metric_test.equal_opportunity_difference())
    print("Test set: Average odds difference = %f" % classified_metric_test.average_odds_difference())
    print("Test set: Theil_index = %f" % classified_metric_test.theil_index())


def print_test_set_classifier_metric(dataset_test, orig_test):
    print("Model: classification metrics")
    classified_metric_test = ClassificationMetric(orig_test,
                                                  dataset_test,
                                                  unprivileged_groups=unprivileged_groups,
                                                  privileged_groups=privileged_groups)
    print("Test set: Classification accuracy = %f" % classified_metric_test.accuracy())
    TPR = classified_metric_test.true_positive_rate()
    TNR = classified_metric_test.true_negative_rate()
    bal_acc_nodebiasing_test = 0.5 * (TPR + TNR)
    print("Test set: Balanced classification accuracy = %f" % bal_acc_nodebiasing_test)
    print("Test set: Disparate impact = %f" % classified_metric_test.disparate_impact())
    print(
        "Test set: Equal opportunity difference = %f" % classified_metric_test.equal_opportunity_difference())
    print("Test set: Average odds difference = %f" % classified_metric_test.average_odds_difference())
    print("Test set: Theil_index = %f" % classified_metric_test.theil_index())


if __name__ == '__main__':
    # dataset_nodebiasing_train, dataset_nodebiasing_test = plain_model.fit_and_pred(dataset_orig_train,
    #                                                                                dataset_orig_test)
    print("model path now is ../org-model/adult/999/test.model adebias as well ")
    # Get the dataset and split into train and test
    dataset_orig = load_preproc_data_adult()
    privileged_groups = [{'sex': 1}]
    unprivileged_groups = [{'sex': 0}]
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.8], shuffle=True)
    sess = tf.Session()
    plain_model = AdversarialDebiasingDnn5(privileged_groups=privileged_groups,
                                           unprivileged_groups=unprivileged_groups,
                                           scope_name='plain_classifier',
                                           debias=False,
                                           sess=sess,
                                           origmodel_path='../org-model-fixnc/adult/29/test.model',
                                           debiasmodel_path='../adebias-model-fixnc/adult/29/test.model')
    dataset_nodebiasing_train = plain_model.predict_with_load_gra(dataset_orig_train)
    dataset_nodebiasing_test = plain_model.predict_with_load_gra(dataset_orig_test)
    print('ok, Plain model - without debiasing - dataset metrics:')
    print_metric(dataset_nodebiasing_train, dataset_nodebiasing_test, orig_test=dataset_orig_test)
    sess.close()
    tf.reset_default_graph()
    sess = tf.Session()

    debiased_model = AdversarialDebiasingDnn5(privileged_groups=privileged_groups,
                                              unprivileged_groups=unprivileged_groups,
                                              scope_name='debiased_classifier',
                                              debias=True,
                                              sess=sess,
                                              origmodel_path='../org-model-fixnc/adult/29/test.model',
                                              debiasmodel_path='../adebias-model-fixnc/adult/29/test.model'
                                              )
    # dataset_debiasing_train_, dataset_debiasing_test_ = debiased_model.fit_and_pred(dataset_orig_train,
    #                                                                                 dataset_orig_test)
    # dataset_debiasing_train = debiased_model.predict_with_load_gra(dataset_orig_train)
    dataset_debiasing_test = debiased_model.predict_with_load_gra(dataset_orig_test)
    print('Model - with debiasing - classification metrics')
    print_test_set_classifier_metric(dataset_debiasing_test, orig_test=dataset_orig_test)
    # print_metric(dataset_debiasing_train, dataset_debiasing_test, orig_test=dataset_orig_test)

