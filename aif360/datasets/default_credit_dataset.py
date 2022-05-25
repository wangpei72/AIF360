import os

import pandas as pd

from aif360.datasets import StandardDataset

default_mappings = {
    'label_maps': [{1.0: 'yes', 0.0: 'no'}],  # 这里需不需要转义字符需要测试一下
    'protected_attribute_maps': [{1.0: [lambda x: x >= 25], 0.0: [lambda x: x < 25]}]
}


class DefaultCreditDataset(StandardDataset):
    """Default of credit card Dataset.
    """

    def __init__(self, label_name='default payment next month', favorable_classes=['yes'],
                 protected_attribute_names=['age'],
                 privileged_classes=[lambda x: x >= 25],
                 instance_weights_name=None,
                 categorical_features=['job', 'marital', 'education', 'default',
                     'housing', 'loan', 'contact', 'month', 'day_of_week',
                     'poutcome'],
                 features_to_keep=[], features_to_drop=[],
                 na_values=["unknown"], custom_preprocessing=None,
                 metadata=default_mappings):
        """See :obj:`StandardDataset` for a description of the arguments.

        By default, this code converts the 'age' attribute to a binary value
        where privileged is `age >= 25` and unprivileged is `age < 25` as in
        :obj:`GermanDataset`.
        """

        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            '..', 'data', 'raw', 'default', 'default_of_credit_card_clients.csv')

        try:
            df = pd.read_csv(filepath, sep=',', na_values=na_values, header=[0],
                         skiprows=[1])
        except IOError as err:
            print("IOError: {}".format(err))
            print("To use this class, please download the following file:")
            print("\n\thttps://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls")
            print("\nplace the files, as-is, in the folder:")
            print("\n\t{}\n".format(os.path.abspath(os.path.join(
               os.path.abspath(__file__), '..', '..', 'data', 'raw', 'default'))))
            import sys
            sys.exit(1)

        super(DefaultCreditDataset, self).__init__(df=df, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)
