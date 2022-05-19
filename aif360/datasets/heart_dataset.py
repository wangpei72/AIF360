import os

import pandas as pd

from aif360.datasets import StandardDataset
# TODO 重写heart dataset类
# presence -有患病可能 absence- 无可能
default_mappings = {
    'label_maps': [{1.0: 'absence', 0.0: 'presence'}],
    'protected_attribute_maps': [{0.0: [lambda x: x >= 54.4], 1.0: [lambda x: x < 54.4]}]
}


class HeartDataset(StandardDataset):
    """Heart disease Dataset.
    """

    def __init__(self, label_name='Probability',
                 favorable_classes=['absence'],
                 protected_attribute_names=['age'],
                 privileged_classes=[lambda x: x < 54.4],
                 instance_weights_name=None,
                 categorical_features=[],
                 features_to_keep=[], features_to_drop=[],
                 na_values=["unknown"], custom_preprocessing=None,
                 metadata=default_mappings):
        """See :obj:`StandardDataset` for a description of the arguments.

        By default, this code converts the 'age' attribute to a binary value
        where privileged is `age <54.4` and unprivileged is `age >= 54.4` as in
        :obj:`HeartDataset`.
        """

        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            '..', 'data', 'raw', 'heart', 'processed.cleveland.data.csv')

        try:
            df = pd.read_csv(filepath, sep=',', na_values=na_values)

        except IOError as err:
            print("IOError: {}".format(err))
            print("To use this class, please download the following file:")
            print("\n\thttps://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/cleveland.data")
            print("\nplace the files, as-is, in the folder:")
            print("\n\t{}\n".format(os.path.abspath(os.path.join(
               os.path.abspath(__file__), '..', '..', 'data', 'raw', 'heart'))))
            import sys
            sys.exit(1)

        super(HeartDataset, self).__init__(df=df, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)
