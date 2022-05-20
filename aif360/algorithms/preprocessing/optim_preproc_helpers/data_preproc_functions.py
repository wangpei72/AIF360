from aif360.datasets import AdultDataset, GermanDataset, \
    CompasDataset, BankDataset, DefaultCreditDataset, \
    HeartDataset, StudentDataset, MEPSDataset19, MEPSDataset20, MEPSDataset21, HomeCreditDataset
from aif360.datasets.helper_script.home_csv_helper import set_OneHotEncoder
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

def load_preproc_data_bank(protected_attributes=None):
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

def load_preproc_data_default(protected_attributes=None):
    def custom_preprocessing(df):

        def group_limit_bal(x):
            if x <= 60000:
                return '<=60k'
            elif x > 60000 and x <= 200000:
                return '60k to 200k'
            elif x > 200000 and x <= 400000:
                return '200k to 400k'
            elif x > 400000:
                return '>400k'
            else:
                return 'others'

        # group limit_bal Age
        # TODO之前曲解了在类中的>25的意思，确实是从》=26开始的年龄，所以之前改的bug（还导致accu下降了）可能要改回去
        df['AGE'] = df['AGE'].apply(lambda x: np.float(x >= 25))
        df['LIMIT_BAL'] = df['LIMIT_BAL'].apply(lambda x: group_limit_bal(x))
        # rename and replace cols
        # df.rename(columns={'default.payment.next.month': 'def_pay'}, inplace=True)
        df['SEX'] = df['SEX'].replace({2: 0.0, 1: 1.0})  # 1 = male while 2= female

        return df

    XD_features = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
                   'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                   'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5','BILL_AMT6',
                   'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
                   ]
    D_features = ['AGE'] if protected_attributes is None else protected_attributes
    Y_features = ['default payment next month']
    X_features = list(set(XD_features) - set(D_features))
    categorical_features = ['LIMIT_BAL']

    # pri classes
    all_privileged_classes = {'AGE': [1.0]}

    # protected attr maps
    all_protected_attribute_maps = {'AGE': {1.0: 'Old', 0.0: 'Young'}}

    dataset_default = DefaultCreditDataset(
        label_name=Y_features[0],
        favorable_classes=[1],
        protected_attribute_names=D_features,
        privileged_classes=[all_privileged_classes[x] for x in D_features],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features+Y_features+D_features,
        metadata={'label_maps': [{1.0: 'default payment', 0.0: 'non-default payment'}],
                  'protected_attribute_maps': [all_protected_attribute_maps[x] for x in D_features]},
        custom_preprocessing=custom_preprocessing)
    return dataset_default



def load_preproc_data_heart(protected_attributes=None):
    def custom_preprocessing(df):
        # 根据平均数划分界限
        df['age'] = df['age'].apply(lambda x: np.float(x < 54.4))

        # binary label col
        df['Probability'] = df['Probability'].apply(lambda x: np.float(x == 0))
        # df['Probability'] = pd.DataFrame.where(df, df['Probability'] == 0, other=1)
        return df

    XD_features = ['age', 'sex', 'cp', 'trestbps', 'chol',
                   'fbs', 'restecg', 'thalach', 'exang',
                   'oldpeak','slope','ca','thal'
                   ]
    D_features = ['age'] if protected_attributes is None else protected_attributes
    Y_features = ['Probability']
    X_features = list(set(XD_features) - set(D_features))
    categorical_features = []

    # pri classes 年轻的人患病可能性较小 属于优势群体 因此上面的lambda表达式给的也是小于号
    # 同时 prob为0 表示没有患病可能的属于优势
    all_privileged_classes = {'age': [1.0]}

    # protected attr maps
    all_protected_attribute_maps = {'age': {1.0: 'Young', 0.0: 'Old'}}

    dataset_heart = HeartDataset(
        label_name=Y_features[0],
        favorable_classes=[1],
        protected_attribute_names=D_features,
        privileged_classes=[all_privileged_classes[x] for x in D_features],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features+Y_features+D_features,
        metadata={'label_maps': [{1.0: 'absence', 0.0: 'presence'}],
                  'protected_attribute_maps': [all_protected_attribute_maps[x] for x in D_features]},
        custom_preprocessing=custom_preprocessing)
    return dataset_heart

def load_preproc_data_student(protected_attributes=None):
    def custom_preprocessing(df):
        # 将所有的binary换成0和1，将所有的numeric维持不变， nominal将进行编码
        df['school'] = df['school'].apply(lambda x: np.float(x == 'GP'))

        df['sex'] = df['sex'].apply(lambda x: np.float(x == 'M'))
        df['address'] = df['address'].apply(lambda x: np.float(x == 'U'))
        df['famsize'] = df['famsize'].apply(lambda x: np.float(x == 'LE3'))
        df['Pstatus'] = df['Pstatus'].apply(lambda x: np.float(x == 'T'))

        df.loc[df['Mjob'] == 'teacher', 'Mjob'] = 1
        df.loc[df['Mjob'] == 'health', 'Mjob'] = 2
        df.loc[df['Mjob'] == 'services', 'Mjob'] = 3
        df.loc[df['Mjob'] == 'at_home', 'Mjob'] = 0
        df.loc[df['Mjob'] == 'other', 'Mjob'] = 0

        df.loc[df['Fjob'] == 'teacher', 'Fjob'] = 1
        df.loc[df['Fjob'] == 'health', 'Fjob'] = 2
        df.loc[df['Fjob'] == 'services', 'Fjob'] = 3
        df.loc[df['Fjob'] == 'at_home', 'Fjob'] = 0
        df.loc[df['Fjob'] == 'other', 'Fjob'] = 0

        df.loc[df['reason'] == 'home', 'reason'] = 1
        df.loc[df['reason'] == 'reputation', 'reason'] = 2
        df.loc[df['reason'] == 'course', 'reason'] = 3
        df.loc[df['reason'] == 'other', 'reason'] = 0

        df.loc[df['guardian'] == 'mother', 'guardian'] = 1
        df.loc[df['guardian'] == 'father', 'guardian'] = 2
        df.loc[df['guardian'] == 'other', 'guardian'] = 0

        df['schoolsup'] = df['schoolsup'].apply(lambda x: np.float(x == 'yes'))
        df['famsup'] = df['famsup'].apply(lambda x: np.float(x == 'yes'))
        df['paid'] = df['paid'].apply(lambda x: np.float(x == 'yes'))
        df['activities'] = df['activities'].apply(lambda x: np.float(x == 'yes'))
        df['nursery'] = df['nursery'].apply(lambda x: np.float(x == 'yes'))
        df['higher'] = df['higher'].apply(lambda x: np.float(x == 'yes'))
        df['internet'] = df['internet'].apply(lambda x: np.float(x == 'yes'))
        df['romantic'] = df['romantic'].apply(lambda x: np.float(x == 'yes'))

        # binary label col
        df['Probability'] = df['Probability'].apply(lambda x: np.float(x >= 11.4))
        # df['Probability'] = pd.DataFrame.where(df, df['Probability'] == 0, other=1)
        return df

    XD_features = ['school','sex','age','address','famsize','Pstatus',
                   'Medu','Fedu','Mjob','Fjob','reason',
                   'guardian','traveltime','studytime','failures','schoolsup',
                   'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet',
                   'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc',
                   'health', 'absences', 'G1', 'G2'
                   ]
    D_features = ['sex'] if protected_attributes is None else protected_attributes
    Y_features = ['Probability']
    X_features = list(set(XD_features) - set(D_features))
    categorical_features = []

    # pri classes 1表示分数大于平均数
    # 优势群体为男性 M
    all_privileged_classes = {'sex': [1.0]}

    # protected attr maps
    all_protected_attribute_maps = {'sex': {1.0: 'Male', 0.0: 'Female'}}

    dataset_student = StudentDataset(
        label_name=Y_features[0],
        favorable_classes=[1],
        protected_attribute_names=D_features,
        privileged_classes=[all_privileged_classes[x] for x in D_features],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features+Y_features+D_features,
        metadata={'label_maps': [{1.0: 'higher than mean', 0.0: 'lower than mean'}],
                  'protected_attribute_maps': [all_protected_attribute_maps[x] for x in D_features]},
        custom_preprocessing=custom_preprocessing)
    return dataset_student

def load_preproc_data_meps19(protected_attributes=None):
    dataset_meps19 = MEPSDataset19()
    return dataset_meps19

def load_preproc_data_meps20(protected_attributes=None):
    dataset_meps20 = MEPSDataset20()
    return dataset_meps20

def load_preproc_data_meps21(protected_attributes=None):
    dataset_meps21 = MEPSDataset21()
    return dataset_meps21

def load_preproc_data_home_credit(protected_attributes=None):
    def custom_preprocessing(df):
        df = df.rename(columns = {'CODE_GENDER' : 'sex'})

        # 将所有的binary换成0和1，将所有的numeric维持不变， nominal将进行编码
        df['NAME_CONTRACT_TYPE'] = df['NAME_CONTRACT_TYPE'].apply(lambda x: np.float(x == 'Cash loans'))
        df['FLAG_OWN_CAR'] = df['FLAG_OWN_CAR'].apply(lambda x: np.float(x == 'Y'))
        df['FLAG_OWN_REALTY'] =df['FLAG_OWN_REALTY'].apply(lambda x: np.float(x == 'Y'))
        df['sex'] = df['sex'].apply(lambda x: np.float(x == 'M'))
        # TODO 14-18 31 需要进行onehot
        # TODO 35 要进行1-7替换 43要替换

        df.loc[df['OCCUPATION_TYPE'] == 'Laborers', 'OCCUPATION_TYPE'] = 1
        df.loc[df['OCCUPATION_TYPE'] == 'Core staff', 'OCCUPATION_TYPE'] = 2
        df.loc[df['OCCUPATION_TYPE'] == 'Accountants', 'OCCUPATION_TYPE'] = 3
        df.loc[df['OCCUPATION_TYPE'] == 'Managers', 'OCCUPATION_TYPE'] = 4
        df.loc[df['OCCUPATION_TYPE'] == 'Drivers', 'OCCUPATION_TYPE'] = 5
        df.loc[df['OCCUPATION_TYPE'] == 'Sales staff', 'OCCUPATION_TYPE'] = 6
        df.loc[df['OCCUPATION_TYPE'] == 'Cleaning staff', 'OCCUPATION_TYPE'] = 7
        df.loc[df['OCCUPATION_TYPE'] == 'Cooking staff', 'OCCUPATION_TYPE'] = 8
        df.loc[df['OCCUPATION_TYPE'] == 'Private service staff', 'OCCUPATION_TYPE'] = 9
        df.loc[df['OCCUPATION_TYPE'] == 'Medicine staff', 'OCCUPATION_TYPE'] = 10
        df.loc[df['OCCUPATION_TYPE'] == 'Security staff', 'OCCUPATION_TYPE'] = 11
        df.loc[df['OCCUPATION_TYPE'] == 'High skill tech staff', 'OCCUPATION_TYPE'] = 12
        df.loc[df['OCCUPATION_TYPE'] == 'Waiters/barmen staff', 'OCCUPATION_TYPE'] = 13
        df.loc[df['OCCUPATION_TYPE'] == 'Low-skill Laborers', 'OCCUPATION_TYPE'] = 14
        df.loc[df['OCCUPATION_TYPE'] == 'Realty agents', 'OCCUPATION_TYPE'] = 15
        df.loc[df['OCCUPATION_TYPE'] == 'Secretaries', 'OCCUPATION_TYPE'] = 16
        df.loc[df['OCCUPATION_TYPE'] == 'IT staff', 'OCCUPATION_TYPE'] = 17
        df.loc[df['OCCUPATION_TYPE'] == 'HR staff', 'OCCUPATION_TYPE'] = 18



        df.loc[df['WEEKDAY_APPR_PROCESS_START'] == 'MONDAY', 'WEEKDAY_APPR_PROCESS_START'] = 1
        df.loc[df['WEEKDAY_APPR_PROCESS_START'] == 'TUESDAY', 'WEEKDAY_APPR_PROCESS_START'] = 2
        df.loc[df['WEEKDAY_APPR_PROCESS_START'] == 'WEDNESDAY', 'WEEKDAY_APPR_PROCESS_START'] = 3
        df.loc[df['WEEKDAY_APPR_PROCESS_START'] == 'THURSDAY', 'WEEKDAY_APPR_PROCESS_START'] = 4
        df.loc[df['WEEKDAY_APPR_PROCESS_START'] == 'FRIDAY', 'WEEKDAY_APPR_PROCESS_START'] = 5
        df.loc[df['WEEKDAY_APPR_PROCESS_START'] == 'SATURDAY', 'WEEKDAY_APPR_PROCESS_START'] = 6
        df.loc[df['WEEKDAY_APPR_PROCESS_START'] == 'SUNDAY', 'WEEKDAY_APPR_PROCESS_START'] = 7

        def group_org_type(x):
            if x in ['Business Entity Type 3']:
                return 1
            elif x in ['XNA']:
                return 0
            else:
                return 2
        df['ORGANIZATION_TYPE'] = df['ORGANIZATION_TYPE'].apply(lambda x: group_org_type(x))
        # df.loc[df['ORGANIZATION_TYPE'] == 'Business Entity Type 3', 'ORGANIZATION_TYPE'] = 1
        # df.loc[df['ORGANIZATION_TYPE'] == 'XNA', 'ORGANIZATION_TYPE'] = 0
        # df.loc[df['OCCUPATION_TYPE'] is str, 'OCCUPATION_TYPE'] = 2


        return df

    XD_features = ['NAME_CONTRACT_TYPE', 'sex', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'NAME_TYPE_SUITE#Children', 'NAME_TYPE_SUITE#Family',
'NAME_TYPE_SUITE#Group of people', 'NAME_TYPE_SUITE#Other_A', 'NAME_TYPE_SUITE#Other_B', 'NAME_TYPE_SUITE#Spouse, partner', 'NAME_TYPE_SUITE#Unaccompanied', 'NAME_TYPE_SUITE#nan', 'NAME_INCOME_TYPE#Businessman',
'NAME_INCOME_TYPE#Commercial associate', 'NAME_INCOME_TYPE#Maternity leave', 'NAME_INCOME_TYPE#Pensioner', 'NAME_INCOME_TYPE#State servant', 'NAME_INCOME_TYPE#Student', 'NAME_INCOME_TYPE#Unemployed', 'NAME_INCOME_TYPE#Working',
'NAME_EDUCATION_TYPE#Academic degree', 'NAME_EDUCATION_TYPE#Higher education', 'NAME_EDUCATION_TYPE#Incomplete higher', 'NAME_EDUCATION_TYPE#Lower secondary', 'NAME_EDUCATION_TYPE#Secondary / secondary special', 'NAME_FAMILY_STATUS#Civil marriage', 'NAME_FAMILY_STATUS#Married',
'NAME_FAMILY_STATUS#Separated', 'NAME_FAMILY_STATUS#Single / not married', 'NAME_FAMILY_STATUS#Unknown', 'NAME_FAMILY_STATUS#Widow', 'NAME_HOUSING_TYPE#Co-op apartment', 'NAME_HOUSING_TYPE#House / apartment', 'NAME_HOUSING_TYPE#Municipal apartment',
'NAME_HOUSING_TYPE#Office apartment', 'NAME_HOUSING_TYPE#Rented apartment', 'NAME_HOUSING_TYPE#With parents', 'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION',
'DAYS_ID_PUBLISH', 'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL',
'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 'WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START', 'REG_REGION_NOT_LIVE_REGION',
'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'ORGANIZATION_TYPE', 'EXT_SOURCE_2',
'EXT_SOURCE_3', 'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_2',
'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9',
'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16',
'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']

    D_features = ['sex'] if protected_attributes is None else protected_attributes
    Y_features = ['TARGET']
    X_features = list(set(XD_features) - set(D_features))
    categorical_features = []

    # pri classes 1表示分数大于平均数
    # 优势群体为男性 M
    all_privileged_classes = {'sex': [1.0]}

    # protected attr maps
    all_protected_attribute_maps = {'sex': {1.0: 'M', 0.0: 'F'}}

    dataset_home_credit = HomeCreditDataset(
        label_name=Y_features[0],
        favorable_classes=[0],
        protected_attribute_names=D_features,
        privileged_classes=[all_privileged_classes[x] for x in D_features],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features+Y_features+D_features,
        metadata={'label_maps': [{1.0: 'reject', 0.0: 'accept'}],
                  'protected_attribute_maps': [all_protected_attribute_maps[x] for x in D_features]},
        custom_preprocessing=custom_preprocessing)
    return dataset_home_credit