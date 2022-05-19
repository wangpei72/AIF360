from aif360.datasets import AdultDataset, GermanDataset, CompasDataset, BankDataset
import pandas as pd
import numpy as np

def load_preproc_data_adult(protected_attributes=None, sub_samp=False, balance=False, convert=True):
    def custom_preprocessing(df):
        """The custom pre-processing function is adapted from
            https://github.com/fair-preprocessing/nips2017/blob/master/Adult/code/Generate_Adult_Data.ipynb
            If sub_samp != False, then return smaller version of dataset truncated to tiny_test data points.
        """

        # Group age by decade
        df['Age (decade)'] = df['age'].apply(lambda x: x//10*10)
        # df['Age (decade)'] = df['age'].apply(lambda x: np.floor(x/10.0)*10.0)

        def group_edu(x):
            if x <= 5:
                return '<6'
            elif x >= 13:
                return '>12'
            else:
                return x

        def age_cut(x):
            if x >= 70:
                return '>=70'
            else:
                return x

        def group_race(x):
            if x == "White":
                return 1.0
            else:
                return 0.0

        # Cluster education and age attributes.
        # Limit education range
        df['Education Years'] = df['education-num'].apply(lambda x: group_edu(x))
        df['Education Years'] = df['Education Years'].astype('category')

        # Limit age range
        df['Age (decade)'] = df['Age (decade)'].apply(lambda x: age_cut(x))

        # Rename income variable
        df['Income Binary'] = df['income-per-year']
        df['Income Binary'] = df['Income Binary'].replace(to_replace='>50K.', value='>50K', regex=True)
        df['Income Binary'] = df['Income Binary'].replace(to_replace='<=50K.', value='<=50K', regex=True)

        # Recode sex and race
        df['sex'] = df['sex'].replace({'Female': 0.0, 'Male': 1.0})
        df['race'] = df['race'].apply(lambda x: group_race(x))

        if sub_samp and not balance:
            df = df.sample(sub_samp)
        if sub_samp and balance:
            df_0 = df[df['Income Binary'] == '<=50K']
            df_1 = df[df['Income Binary'] == '>50K']
            df_0 = df_0.sample(int(sub_samp/2))
            df_1 = df_1.sample(int(sub_samp/2))
            df = pd.concat([df_0, df_1])
        return df

    XD_features = ['Age (decade)', 'Education Years', 'sex', 'race']
    D_features = ['sex', 'race'] if protected_attributes is None else protected_attributes
    Y_features = ['Income Binary']
    X_features = list(set(XD_features)-set(D_features))
    categorical_features = ['Age (decade)', 'Education Years']

    # privileged classes
    all_privileged_classes = {"sex": [1.0],
                              "race": [1.0]}

    # protected attribute maps
    all_protected_attribute_maps = {"sex": {1.0: 'Male', 0.0: 'Female'},
                                    "race": {1.0: 'White', 0.0: 'Non-white'}}
    dataset_adult = AdultDataset(
        label_name=Y_features[0],
        favorable_classes=['>50K', '>50K.'],
        protected_attribute_names=D_features,
        privileged_classes=[all_privileged_classes[x] for x in D_features],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features+Y_features+D_features,
        na_values=['?'],
        metadata={'label_maps': [{1.0: '>50K', 0.0: '<=50K'}],
                  'protected_attribute_maps': [all_protected_attribute_maps[x]
                                for x in D_features]},
        custom_preprocessing=custom_preprocessing)

    return dataset_adult


def convert_two_dims_labels(dataset):
    temp = []
    for i in dataset.labels:
        if i == dataset.unfavorable_label:
            temp.append([1.0, 0.0])
        else:
            temp.append([0.0, 1.0])
    arr = np.array(temp, dtype=np.float64)
    return arr


def load_preproc_data_compas(protected_attributes=None):
    def custom_preprocessing(df):
        """The custom pre-processing function is adapted from
            https://github.com/fair-preprocessing/nips2017/blob/master/compas/code/Generate_Compas_Data.ipynb
        """

        df = df[['age', 'c_charge_degree', 'race', 'age_cat', 'score_text',
                 'sex', 'priors_count', 'days_b_screening_arrest', 'decile_score',
                 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]

        # Indices of data samples to keep
        ix = df['days_b_screening_arrest'] <= 30
        ix = (df['days_b_screening_arrest'] >= -30) & ix
        ix = (df['is_recid'] != -1) & ix
        ix = (df['c_charge_degree'] != "O") & ix
        ix = (df['score_text'] != 'N/A') & ix
        df = df.loc[ix,:]
        df['length_of_stay'] = (pd.to_datetime(df['c_jail_out'])-
                                pd.to_datetime(df['c_jail_in'])).apply(
                                                        lambda x: x.days)

        # Restrict races to African-American and Caucasian
        dfcut = df.loc[~df['race'].isin(['Native American','Hispanic','Asian','Other']),:]

        # Restrict the features to use
        dfcutQ = dfcut[['sex','race','age_cat','c_charge_degree','score_text','priors_count','is_recid',
                'two_year_recid','length_of_stay']].copy()

        # Quantize priors count between 0, 1-3, and >3
        def quantizePrior(x):
            if x <=0:
                return '0'
            elif 1<=x<=3:
                return '1 to 3'
            else:
                return 'More than 3'

        # Quantize length of stay
        def quantizeLOS(x):
            if x<= 7:
                return '<week'
            if 8<x<=93:
                return '<3months'
            else:
                return '>3 months'

        # Quantize length of stay
        def adjustAge(x):
            if x == '25 - 45':
                return '25 to 45'
            else:
                return x

        # Quantize score_text to MediumHigh
        def quantizeScore(x):
            if (x == 'High')| (x == 'Medium'):
                return 'MediumHigh'
            else:
                return x

        def group_race(x):
            if x == "Caucasian":
                return 1.0
            else:
                return 0.0

        dfcutQ['priors_count'] = dfcutQ['priors_count'].apply(lambda x: quantizePrior(x))
        dfcutQ['length_of_stay'] = dfcutQ['length_of_stay'].apply(lambda x: quantizeLOS(x))
        dfcutQ['score_text'] = dfcutQ['score_text'].apply(lambda x: quantizeScore(x))
        dfcutQ['age_cat'] = dfcutQ['age_cat'].apply(lambda x: adjustAge(x))

        # Recode sex and race
        dfcutQ['sex'] = dfcutQ['sex'].replace({'Female': 1.0, 'Male': 0.0})
        dfcutQ['race'] = dfcutQ['race'].apply(lambda x: group_race(x))

        features = ['two_year_recid',
                    'sex', 'race',
                    'age_cat', 'priors_count', 'c_charge_degree']

        # Pass vallue to df
        df = dfcutQ[features]

        return df

    XD_features = ['age_cat', 'c_charge_degree', 'priors_count', 'sex', 'race']
    D_features = ['sex', 'race']  if protected_attributes is None else protected_attributes
    Y_features = ['two_year_recid']
    X_features = list(set(XD_features)-set(D_features))
    categorical_features = ['age_cat', 'priors_count', 'c_charge_degree']

    # privileged classes
    all_privileged_classes = {"sex": [1.0],
                              "race": [1.0]}

    # protected attribute maps
    all_protected_attribute_maps = {"sex": {0.0: 'Male', 1.0: 'Female'},
                                    "race": {1.0: 'Caucasian', 0.0: 'Not Caucasian'}}


    return CompasDataset(
        label_name=Y_features[0],
        favorable_classes=[0],
        protected_attribute_names=D_features,
        privileged_classes=[all_privileged_classes[x] for x in D_features],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features+Y_features+D_features,
        na_values=[],
        metadata={'label_maps': [{1.0: 'Did recid.', 0.0: 'No recid.'}],
                  'protected_attribute_maps': [all_protected_attribute_maps[x]
                                for x in D_features]},
        custom_preprocessing=custom_preprocessing)

def load_preproc_data_german(protected_attributes=None):
    """
    Load and pre-process german credit dataset.
    Args:
        protected_attributes(list or None): If None use all possible protected
            attributes, else subset the protected attributes to the list.

    Returns:
        GermanDataset: An instance of GermanDataset with required pre-processing.

    """
    def custom_preprocessing(df):
        """ Custom pre-processing for German Credit Data
        """

        def group_credit_hist(x):
            if x in ['A30', 'A31', 'A32']:
                return 'None/Paid'
            elif x == 'A33':
                return 'Delay'
            elif x == 'A34':
                return 'Other'
            else:
                return 'NA'

        def group_employ(x):
            if x == 'A71':
                return 'Unemployed'
            elif x in ['A72', 'A73']:
                return '1-4 years'
            elif x in ['A74', 'A75']:
                return '4+ years'
            else:
                return 'NA'

        def group_savings(x):
            if x in ['A61', 'A62']:
                return '<500'
            elif x in ['A63', 'A64']:
                return '500+'
            elif x == 'A65':
                return 'Unknown/None'
            else:
                return 'NA'

        def group_status(x):
            if x in ['A11', 'A12']:
                return '<200'
            elif x in ['A13']:
                return '200+'
            elif x == 'A14':
                return 'None'
            else:
                return 'NA'

        status_map = {'A91': 1.0, 'A93': 1.0, 'A94': 1.0,
                    'A92': 0.0, 'A95': 0.0}
        df['sex'] = df['personal_status'].replace(status_map)


        # group credit history, savings, and employment
        df['credit_history'] = df['credit_history'].apply(lambda x: group_credit_hist(x))
        df['savings'] = df['savings'].apply(lambda x: group_savings(x))
        df['employment'] = df['employment'].apply(lambda x: group_employ(x))
        df['age'] = df['age'].apply(lambda x: np.float(x >= 26))
        df['status'] = df['status'].apply(lambda x: group_status(x))

        return df

    # Feature partitions
    XD_features = ['credit_history', 'savings', 'employment', 'sex', 'age']
    D_features = ['sex', 'age'] if protected_attributes is None else protected_attributes
    Y_features = ['credit']
    X_features = list(set(XD_features)-set(D_features))
    categorical_features = ['credit_history', 'savings', 'employment']

    # privileged classes
    all_privileged_classes = {"sex": [1.0],
                              "age": [1.0]}

    # protected attribute maps
    all_protected_attribute_maps = {"sex": {1.0: 'Male', 0.0: 'Female'},
                                    "age": {1.0: 'Old', 0.0: 'Young'}}

    return GermanDataset(
        label_name=Y_features[0],
        favorable_classes=[1],
        protected_attribute_names=D_features,
        privileged_classes=[all_privileged_classes[x] for x in D_features],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features+Y_features+D_features,
        metadata={ 'label_maps': [{1.0: 'Good Credit', 2.0: 'Bad Credit'}],
                   'protected_attribute_maps': [all_protected_attribute_maps[x]
                                for x in D_features]},
        custom_preprocessing=custom_preprocessing)

def load_preproc_data_bank(protected_atrributes=None):
    def custom_preprocessing(df):
        # group age by decade
        df['Age (decade)'] = df['age'].apply(lambda x: x//10*10)

        def group_job(x):
            if x in ['admin.', 'blue-collar', 'entrepreneur',
                     'housemaid', 'management', 'self-employed',
                     'services', 'technician']:
                return 'employed'
            elif x in ['student', 'unemployed']:
                return 'unemployed'
            elif x in ['retired']:
                return 'retired'
            else:
                return 'unknown'

        def group_marital(x):
            if x in ['married']:
                return 'married'
            else:
                return 'others'

        def group_edu(x):
            if x in ['illiterate', 'basic.4y', 'basic.6y']:
                return '<9'
            elif x in ['basic.9y', 'high.school']:
                return '9-12'
            elif x in ['university.degree', 'professional.course']:
                return '>12'
            else:
                return 'unknown'

        def group_campaign(x):
            if x <= 1:
                return '<=1'
            else:
                return '>1'

        def group_pdays(x):
            if x == 999:
                return 'non-contacted'
            else:
                return 'contacted>once'

        def group_previous(x):
            if x == 0:
                return 'never'
            else:
                return 'ever'

        def group_poutcome(x):
            if x in ['failure', 'nonexistent']:
                return 0.0
            elif x in ['success']:
                return 1.0

        def age_cut(x):
            if x >= 70:
                return '>=70'
            else:
                return x

        # 补充5-7的group
        def group_default(x):
            if x in ['yes']:
                return 'yes'
            elif x in ['no']:
                return 'no'
            else:
                return 'unknown'

        def group_housing(x):
            if x in ['yes']:
                return 'yes'
            elif x in ['no']:
                return 'no'
            else:
                return 'unknown'

        def group_loan(x):
            if x in ['yes']:
                return 'yes'
            elif x in ['no']:
                return 'no'
            else:
                return 'unknown'

        df['Education Years'] = df['education'].apply(lambda x: group_edu(x))
        df['Education Years'] = df['Education Years'].astype('category')

        df['Age (decade)'] = df['Age (decade)'].apply(lambda x: age_cut(x))

        # df['Subscribe Prob'] = df['y']
        # df['Subscribe Prob'] = df['Subscribe Prob'].replace()

        # group job marital campaign pdays previous poutcome
        df['age'] = df['age'].apply(lambda x: np.float(x >= 25))
        # 注意加上了这行并且在下面使用 相当于之前的Age decade被弃用

        df['default'] = df['default'].apply(lambda x: group_default(x))
        df['housing'] = df['housing'].apply(lambda x: group_housing(x))
        df['loan'] = df['loan'].apply(lambda x: group_loan(x))
        df['job'] = df['job'].apply(lambda x: group_job(x))
        df['marital'] = df['marital'].apply(lambda x: group_marital(x))
        df['campaign'] = df['campaign'].apply(lambda x: group_campaign(x))
        df['pdays'] = df['pdays'].apply(lambda x: group_pdays(x))
        df['previous'] = df['previous'].apply(lambda x: group_previous(x))
        df['poutcome'] = df['poutcome'].apply(lambda x: group_poutcome(x))

        # TODO default开始 5-11的属性的categorical属性没有做group，12-15是数字的反而做了，可能是不需要的？

        return df

    # 特征分离 partitions
    XD_features = ['age', 'job', 'marital', 'Education Years',
                   'default', 'housing', 'loan',
                   'campaign', 'pdays', 'previous',
                   'poutcome']
    D_features = ['age']
    Y_features = ['y']
    X_features = list(set(XD_features) - set(D_features))
    categorical_features = ['job', 'marital', 'Education Years',
                            'default', 'housing', 'loan', 'campaign', 'pdays', 'previous',
                            'poutcome']

    # privileged classes 优势类别
    all_privileged_classes = {"age": [1.0]}

    # 敏感属性映射
    all_protected_attribute_maps = {'age': {1.0: 'Old', 0.0: 'Young'}}


    dataset_bank = BankDataset(
        label_name=Y_features[0],
        favorable_classes=['yes'],
        protected_attribute_names=D_features,
        privileged_classes=[all_privileged_classes[x] for x in D_features],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features+Y_features+D_features,
        metadata={ 'label_maps': [{1.0: 'yes', 0.0: 'no'}],
                   'protected_attribute_maps': [all_protected_attribute_maps[x]
                                                for x in D_features]},
        custom_preprocessing=custom_preprocessing)

    return dataset_bank
