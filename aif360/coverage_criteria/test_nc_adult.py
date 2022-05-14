
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import gc

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

# sys.path.append("D:\\wp\\PycharmProjects\\AIF360\\aif360\\algorithms\\inprocessing")
sys.path.append("../")


from aif360.load_model.network import *
from aif360.load_model.layer import *



from aif360.coverage_criteria.utils import init_coverage_tables, neuron_covered, update_coverage, \
   get_single_full_test_sample
from aif360.coverage_criteria import wrt_xls

FLAGS = flags.FLAGS
# from aif360.datasets import BinaryLabelDataset
# from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
# from aif360.metrics import BinaryLabelDatasetMetric
# from aif360.metrics import ClassificationMetric
# from aif360.metrics.utils import compute_boolean_conditioning_vector
#
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_compas, load_preproc_data_german

from aif360.algorithms.inprocessing.adversarial_debiasing_dnn5 import AdversarialDebiasingDnn5




def dnn5(input_shape=(None, 18), nb_classes=2):
    """
    The implementation of a DNN model
    :param input_shape: the shape of dataset
    :param nb_classes: the number of classes
    :return: a DNN model
    """
    activation = ReLU
    layers = [Linear(64),
              activation(),
              Linear(32),
              activation(),
              Linear(16),
              activation(),
              Linear(8),
              activation(),
              Linear(4),
              activation(),
              Linear(nb_classes),
              Softmax()]

    model = MLP(layers, input_shape)
    return model


def model_load(datasets):
    tf.reset_default_graph()
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.7
    # config.gpu_options.allow_growth = True
    # Create TF session and set as Keras backend session
    sess = tf.Session(config=config)
    print("Created TensorFlow session.")
    dataset_orig = load_preproc_data_adult()

    privileged_groups = [{'sex': 1}]
    unprivileged_groups = [{'sex': 0}]

    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.8], shuffle=True)
    plain_model = AdversarialDebiasingDnn5(privileged_groups=privileged_groups,
                                           unprivileged_groups=unprivileged_groups,
                                           scope_name='plain_classifier',
                                           debias=False,
                                           sess=sess)
    plain_model.fit_without_train(dataset_orig_test)
    input_shape = (None, 18)
    nb_classes = 1
    x = tf.placeholder(tf.float32, shape=input_shape)
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    feed_dict = None
    # TODO 这里要修改
    # model = dnn5(input_shape, nb_classes)
    #
    # preds = model(x)

    print("Defined TensorFlow model graph.")

    # saver = tf.train.Saver()

    model_path = '../../model/adversarial-debiasing/adebias-model-fix/adult/999/test.model'
    saver = tf.train.import_meta_graph(model_path + '.meta')

    saver.restore(sess, model_path)

    return sess, plain_model.preds_symbolic_output, x, y, plain_model.classifier_model, feed_dict


def neuron_coverage(
                    datasets, model_name, de=False, attack='fgsm'):
    """
    :param datasets
    :param model
    :param samples_path
    :return:
    """
    tuple_res =  get_single_full_test_sample()
    samples = tuple_res[2]
    n_batches = 10
    X_train_boundary = tuple_res[0]


    for i in range(n_batches):
        print(i)

        tf.reset_default_graph()

        sess, preds, x, y, model, feed_dict = model_load(datasets=datasets)
        model_layer_dict = init_coverage_tables(model)
        model_layer_dict = update_coverage(sess, x, samples, model, model_layer_dict, feed_dict, threshold=0)
        sess.close()
        del sess, preds, x, y, model, feed_dict
        gc.collect()

        result = neuron_covered(model_layer_dict)[2]
        print('covered neurons percentage %d neurons %f'
              % (len(model_layer_dict), result))
        return result


def main(argv=None):
    store_path = "../coverage-result/dnn5/adver-adult-debiased/"
    nc_to_save = []

    nc_res = neuron_coverage(datasets=FLAGS.datasets,
                    model_name=FLAGS.model,
                    )
    print("neuron_coverage returns %f" % nc_res)
    nc_to_save.append(nc_res)

    nc_to_save = np.array(nc_to_save, dtype=np.float32)
    np.save(store_path + 'nc.npy', nc_to_save)

# TODO xls file writer


if __name__ == '__main__':
    flags.DEFINE_string('datasets', 'adult', 'The target datasets.')
    flags.DEFINE_string('model', 'dnn5', 'The name of model')

    tf.app.run()
