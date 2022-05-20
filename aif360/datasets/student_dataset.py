import os

import pandas as pd

from aif360.datasets import StandardDataset
# TODO 重写Student dataset类

default_mappings = {
    'label_maps': [{1.0: 'higher than mean', 0.0: 'lower than mean'}],
    'protected_attribute_maps': [{1.0: 'M', 0.0: 'F'}]
}


class StudentDataset(StandardDataset):
    """Student performance Dataset.
    """

    def __init__(self, label_name='Probability',
                 favorable_classes=['M'],
                 protected_attribute_names=['age'],
                 privileged_classes=[lambda x: x >= 11.4],
                 instance_weights_name=None,
                 categorical_features=[],
                 features_to_keep=[], features_to_drop=[],
                 na_values=["unknown"], custom_preprocessing=None,
                 metadata=default_mappings):
        """See :obj:`StandardDataset` for a description of the arguments.

        By default, this code converts the 'age' attribute to a binary value
        where privileged is `age <54.4` and unprivileged is `age >= 54.4` as in
        :obj:`StudentDataset`.
        """

        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            '..', 'data', 'raw', 'student', 'Student.csv')

        try:
            df = pd.read_csv(filepath, sep=',', na_values=na_values)

        except IOError as err:
            print("IOError: {}".format(err))
            print("To use this class, please download the following file:")
            print("\n\thttps://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip")
            print("\nunzip the file and place the files, as-is, in the folder:")
            print("\n\t{}\n".format(os.path.abspath(os.path.join(
               os.path.abspath(__file__), '..', '..', 'data', 'raw', 'student'))))
            import sys
            sys.exit(1)

        super(StudentDataset, self).__init__(df=df, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)
