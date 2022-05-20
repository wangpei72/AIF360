import os

import pandas as pd

from aif360.datasets import StandardDataset
# TODO 重写HomeCredit dataset类
# Target is 用户是否有还款困难 有是1 没有困难是0
default_mappings = {
    'label_maps': [{1.0: 'reject', 0.0: 'accept'}],
    'protected_attribute_maps': [{1.0: 'M', 0.0: 'F'}]
}


class HomeCreditDataset(StandardDataset):
    """Home Credit  Dataset.
    """

    def __init__(self, label_name='TARGET',
                 favorable_classes=[0],
                 protected_attribute_names=['sex'],
                 privileged_classes=['M'],
                 instance_weights_name=None,
                 categorical_features=[],
                 features_to_keep=[], features_to_drop=[],
                 na_values=[""], custom_preprocessing=None,
                 metadata=default_mappings):
        """See :obj:`StandardDataset` for a description of the arguments.

        By default, this code converts the 'sex' attribute to a binary value
        where privileged is `M` and unprivileged is `F` as in
        :obj:`HomeCreditDataset`.
        """

        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            '..', 'data', 'raw', 'home', 'home_onehot_12_16.csv')

        try:
            df = pd.read_csv(filepath, sep=',', na_values=na_values)

        except IOError as err:
            print("IOError: {}".format(err))
            print("To use this class, please download the following file:")
            print("\n\thttps://www.kaggle.com/c/home-credit-default-risk")
            print("\nunzip the file and place the files, as-is, in the folder:")
            print("\n\t{}\n".format(os.path.abspath(os.path.join(
               os.path.abspath(__file__), '..', '..', 'data', 'raw', 'home'))))
            import sys
            sys.exit(1)

        super(HomeCreditDataset, self).__init__(df=df, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)
