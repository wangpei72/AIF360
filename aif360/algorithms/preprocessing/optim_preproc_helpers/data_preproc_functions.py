from aif360.datasets import AdultDataset, GermanDataset, \
    CompasDataset, BankDataset, DefaultCreditDataset, \
    HeartDataset, StudentDataset, MEPSDataset19, MEPSDataset20, MEPSDataset21, HomeCreditDataset
from aif360.datasets.helper_script.home_csv_helper import set_OneHotEncoder
import pandas as pd
import numpy as np
from . import helper as hp
from . import compas_helper as chp
from . import bank_helper as bhp
from . import heart_helper as hhp
from . import student_helper as shp
from . import default_credit_helper as dchp

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

    Y_features = ['income-per-year']
    skip_feat = ['sex', 'race', 'education-num']
    label_maps= {1: '>50K', 0: '<=50K'}
    def custom_preprocessing_new(df):
        # 1先找到标签 Y_features 转换成二分的，可能有些str要replace
        df['income-per-year'] = df['income-per-year'].replace(to_replace='>50K.', value='>50K', regex=True)
        df['income-per-year'] = df['income-per-year'].replace(to_replace='<=50K.', value='<=50K', regex=True)
        # 2再找到敏感属性，进行二分的处理, 同时找到age属性，所有的age属性按照age思路再workflow进行处理，原先的处理都废弃
        # 3 找到skip feat，包括的种类有 敏感属性，可以直接保留的属性例如edu yrs
        df['sex'] = df['sex'].replace({'Female': 0, 'Male': 1})
        df['race'] = df['race'].apply(lambda x: np.int(x == "White"))
        df['income-per-year'] = df['income-per-year'].apply(lambda x: np.int(x == '>50K'))
        # 3逐一观察numric的列，如果是小的无规则的列归一化为1-9;大的1-99;较为离散的二分成0-1
        #  str类型不用管，会自动编码成数字
        df = hp.work_flow(df, Y_features, skip_feat=skip_feat,
                          binary_0_feat=['capital-gain', 'capital-loss'], age_feat='age')
        hp.wrt_descrip_txt(df, 'adult', Y_feat=Y_features, D_feat=D_features,
                           Y_map=label_maps, D_map=all_protected_attribute_maps,P_map=all_privileged_classes)
        return df
    XD_features = ['age','workclass','fnlwgt','education','education-num','marital-status',
                   'occupation','relationship','race','personal_status','capital-gain',
                   'capital-loss','hours-per-week','native-country']
    D_features = ['personal_status', 'race'] if protected_attributes is None else protected_attributes
    # Y_features = ['income-per-year']
    X_features = list(set(XD_features)-set(D_features))
    categorical_features = []

    # privileged classes
    all_privileged_classes = {'personal_status': [1],
                              "race": [1]}

    # protected attribute maps
    all_protected_attribute_maps = {'personal_status': {1: 'Male', 0: 'Female'},
                                    "race": {1: 'White', 0: 'Non-white'}}
    #  protected_attribute_names=D_features,
    #         privileged_classes=[all_privileged_classes[x] for x in D_features],
    dataset_adult = AdultDataset(
        label_name=Y_features[0],
        favorable_classes=[1],
        protected_attribute_names=[],
        privileged_classes=[],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features+Y_features+D_features,
        na_values=['?'],
        metadata={'label_maps': [{1: '>50K', 0: '<=50K'}],
                  'protected_attribute_maps': [all_protected_attribute_maps[x]
                                for x in D_features]},
        custom_preprocessing=custom_preprocessing_new)

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

    XD_features = [
        'name', 'first', 'last', 'sex', 'dob', 'age', 'age_cat', 'race', 'juv_fel_count'
        , 'decile_score', 'juv_misd_count', 'juv_other_count', 'priors_count', 'days_b_screening_arrest', 'c_jail_in',
        'c_jail_out', 'c_case_number', 'c_offense_date', 'c_arrest_date'
        , 'c_days_from_compas', 'c_charge_degree', 'c_charge_desc', 'is_recid', 'r_case_number', 'r_charge_degree',
        'r_days_from_arrest', 'r_offense_date', 'r_charge_desc', 'r_jail_in'
        , 'r_jail_out', 'violent_recid', 'is_violent_recid', 'vr_case_number', 'vr_charge_degree', 'vr_offense_date',
        'vr_charge_desc', 'type_of_assessment', 'decile_score.1', 'score_text'
        , 'screening_date', 'v_type_of_assessment', 'v_decile_score', 'v_score_text', 'v_screening_date', 'in_custody',
        'out_custody', 'priors_count.1', 'start', 'end'
        , 'event'
    ]
    drop = ['id', 'name', 'first', 'last', 'compas_screening_date', 'r_case_number', 'dob',
            'age_cat', 'c_case_number', 'c_offense_date', 'c_arrest_date', 'r_charge_degree',
            'r_case_number', 'r_days_from_arrest',
            'r_offense_date', 'violent_recid', 'vr_case_number',
            'vr_charge_degree', 'vr_offense_date', 'vr_charge_desc', 'screening_date',
            'v_screening_date', 'v_type_of_assessment', 'type_of_assessment', 'priors_count.1', 'decile_score.1']
    D_features = ['sex', 'race']
    X_features = list(set(XD_features) - set(drop) - set(D_features))
    Y_features = ['two_year_recid']  # 优势为label ： 0
    all_privileged_classes = {"sex": [1],
                              "race": [1]}
    # protected attribute maps
    all_protected_attribute_maps = {"sex": {0: 'Male', 1: 'Female'},
                                    "race": {1: 'Caucasian', 0: 'Not Caucasian'}}
    label_maps = {1.0: 'Did recid.', 0.0: 'No recid.'}  # 优势label是 no-recid 无再犯
    features_to_drop = ['compas_screening_date']
    categorical_features = []
    def custom_preprocessing_new(df):
        df.drop([ 'name', 'first', 'last', 'compas_screening_date', 'r_case_number', 'dob',
                 'age_cat', 'c_case_number', 'c_offense_date', 'c_arrest_date', 'r_charge_degree',
                 'r_case_number', 'r_days_from_arrest',
                 'r_offense_date', 'violent_recid', 'vr_case_number',
                 'vr_charge_degree', 'vr_offense_date', 'vr_charge_desc', 'screening_date',
                 'v_screening_date', 'v_type_of_assessment', 'type_of_assessment',
                 'priors_count.1', 'decile_score.1'], axis=1, inplace=True)
        # df = df.dropna()
        # 进行日期的计算之前将所有日期都是na的行drop掉
        df.dropna(subset=['out_custody', 'in_custody',
                          'r_jail_in', 'r_jail_out',
                          'c_jail_in', 'c_jail_out'], how='all', inplace=True)

        newly_added_days_feat = ['length_of_stay', 'length_of_stay.1', 'length_of_stay.2']
        df['length_of_stay'] = (pd.to_datetime(df['c_jail_out']) -
                                pd.to_datetime(df['c_jail_in'])).apply(
            lambda x: x.days)
        df['length_of_stay.1'] = (pd.to_datetime(df['r_jail_out']) -
                                  pd.to_datetime(df['r_jail_in'])).apply(
            lambda x: x.days)
        df['length_of_stay.2'] = (pd.to_datetime(df['out_custody']) -
                                  pd.to_datetime(df['in_custody'])).apply(
            lambda x: x.days)
        df['start_end'] = df['start'].astype('int32') - df['end'].astype('int32')
        df['max_duration'] = df[['length_of_stay', 'length_of_stay.1', 'length_of_stay.2']].max(axis=1)
        # df = do_description(df)
        df.drop(['out_custody', 'in_custody',
                 'r_jail_in', 'r_jail_out',
                 'c_jail_in', 'c_jail_out',
                 'start', 'end',
                 'length_of_stay', 'length_of_stay.1', 'length_of_stay.2'], axis=1, inplace=True)
        df['sex'] = df['sex'].replace({'Female': 1, 'Male': 0})
        df['race'] = df['race'].apply(lambda x: np.int(x == 'Caucasian'))
        skip_feat = ['sex', 'race', 'juv_fel_count', 'decile_score', 'juv_misd_count', 'juv_other_count',
                     'priors_count', 'is_recid', 'is_violent_recid', 'decile_score.1', 'v_decile_score',
                     'priors_count.1', 'event']
        norm_0_99 = ['start', 'end', 'max_duration', 'start_end']
        norm_0_9 = []
        c_days = ['c_days_from_compas']
        days_b = ['days_b_screening_arrest']
        age_div_10 = ['age']

        df = chp.work_flow(df, y_labels=Y_features, skip_feat=skip_feat, age_div_10=age_div_10,
                       norm_0_99=norm_0_99, norm_0_9=norm_0_9, days=newly_added_days_feat, c_days_from_compas=c_days,
                       days_b_screening_arrest=days_b)
        # df.to_csv('df_compas.csv')
        hp.wrt_descrip_txt(df, 'compas', Y_feat=Y_features, D_feat=D_features, Y_map=label_maps
                               , D_map=all_protected_attribute_maps, P_map=all_privileged_classes)
        return df
    # 被清零的privileged_classes [all_privileged_classes[x] for x in D_features]
    return CompasDataset(
        label_name=Y_features[0],
        favorable_classes=[0],
        protected_attribute_names=[],
        privileged_classes=[],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features+Y_features+D_features,
        na_values=[],
        metadata={'label_maps': [{1.0: 'Did recid.', 0.0: 'No recid.'}],
                  'protected_attribute_maps': [all_protected_attribute_maps[x]
                                for x in D_features]},
        custom_preprocessing=custom_preprocessing_new)


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
        df['credit_history'] = df['credit_history'].apply(lambda x: group_credit_hist(x))
        df['savings'] = df['savings'].apply(lambda x: group_savings(x))
        df['employment'] = df['employment'].apply(lambda x: group_employ(x))
        df['age'] = df['age'].apply(lambda x: np.float(x >= 26))
        df['status'] = df['status'].apply(lambda x: group_status(x))

        return df
    categorical_features = []

    # privileged classes
    all_privileged_classes = {'personal_status': [1],
                              "age": lambda x: x > 2.5} # 实际逻辑是 30往上才算old

    # protected attribute maps
    all_protected_attribute_maps = {'personal_status': {1: 'Male', 0: 'Female'},
                                    "age": {1: 'Old', 0: 'Young'}}
    label_maps = {1: 'Good Credit', 0: 'Bad Credit'}
    XD_features = ['status','month','credit_history','purpose','credit_amount','savings','employment',\
                  'investment_as_income_percentage','personal_status','other_debtors',\
                'residence_since','property','age','installment_plans','housing','number_of_credits',\
                'skill_level','people_liable_for','telephone','foreign_worker']
    D_features = ['personal_status', 'age'] if protected_attributes is None else protected_attributes

    X_features = list(set(XD_features)-set(D_features))
    Y_features = ['credit']
    skip_feat = ['personal_status', 'investment_as_income_percentage', 'residence_since',
                 'number_of_credits', 'people_liable_for']
    norm_1_99 = ['credit_amount']
    age_div_10 = ['age']
    def custom_preprocessing_new(df):
        """ Custom pre-processing for German Credit Data
        """
        #label: 1 - good 0 - bad
        df['credit'] = df['credit'].apply(lambda x: np.int(x == 1))
        # pro attr: 'personal_status' (sex) 1- male 0-female
        status_map = {'A91': 1, 'A93': 1, 'A94': 1,
                      'A92': 0, 'A95': 0}
        df['personal_status'] = df['personal_status'].replace(status_map)
        # df.rename(columns={'personal_status': 'sex'}, inplace=True)
        df = hp.work_flow(df, y_labels=Y_features, skip_feat=skip_feat, age_div_10=age_div_10,
                          norm_1_99=norm_1_99
                          )
        hp.wrt_descrip_txt(df, 'german', Y_features, D_features, label_maps, all_protected_attribute_maps, all_privileged_classes)
        return df
    # TODO 进行numpy导出的时候，不做pri的lambda防止数据被更改为0-1的
    return GermanDataset(
        label_name=Y_features[0],
        favorable_classes=[1],
        protected_attribute_names=[],
        privileged_classes=[],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features+Y_features+D_features,
        metadata={ 'label_maps': [{1: 'Good Credit', 0: 'Bad Credit'}],
                   'protected_attribute_maps': [all_protected_attribute_maps[x]
                                for x in D_features]},
        custom_preprocessing=custom_preprocessing_new)

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
    drop = []
    D_features = ['age']
    Y_features = ['y']
    X_features = list(set(XD_features) - set(drop) - set(D_features))
    categorical_features = ['job', 'marital', 'Education Years',
                                'default', 'housing', 'loan', 'campaign', 'pdays', 'previous',
                                'poutcome']

    # privileged classes 优势类别
    all_privileged_classes = {"age": [1.0]}

    # 敏感属性映射
    all_protected_attribute_maps = {'age': {1.0: 'Old', 0.0: 'Young'}}
    label_maps = {1.0: 'yes', 0.0: 'no'}
    r_y_map = {'yes': 1, 'no': 0}

    def custom_preprocessing_new(df):
        # df = pd.read_csv('../../../data/raw/bank/bank-additional.csv', sep=';', na_values="unknown")
        # 1 dropna不用特地做
        # 2 敏感属性优劣势处理 标签优劣势处理 特判的列表例如age
        age_div_10 = ['age']
        df["y"] = df["y"].replace(r_y_map)
        # 3 进行numeric归一化方式的判断，整理出list，传入workflow参数
        int_cas = [
            "campaign", "previous"
        ]
        norm_0_99 = ['start', 'end', 'max_duration', 'start_end']
        bin_0 = ["emp.var.rate"]
        bin_4_5 = ["euribor3m"]
        bin_93 = ["cons.price.idx"]
        bin_999 = ["pdays"]

        df = bhp.work_flow(df, y_labels=Y_features, int_cas=int_cas, age_div_10=age_div_10, bin_0=bin_0,
                       bin_4_5=bin_4_5, bin_93=bin_93, bin_999=bin_999)
        hp.wrt_descrip_txt(df, 'bank', Y_feat=Y_features, D_feat=D_features,
                               Y_map=label_maps, D_map=all_protected_attribute_maps,
                               P_map=all_privileged_classes)
        return df
    # 为了防止奇怪行为，导出npy时候这两个参数被删掉了
    # protected_attribute_names=D_features,
    #         privileged_classes=[all_privileged_classes[x] for x in D_features],
    dataset_bank = BankDataset(
        label_name=Y_features[0],
        favorable_classes=[1],
        protected_attribute_names=[],
        privileged_classes=[],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features+Y_features+D_features,
        metadata={ 'label_maps': [{1.0: 'yes', 0.0: 'no'}],
                   'protected_attribute_maps': [all_protected_attribute_maps[x]
                                                for x in D_features]},
        custom_preprocessing=custom_preprocessing_new)

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
    def custom_preprocessing_new(df):
        # df = pd.read_csv('../../../data/raw/default/default_of_credit_card_clients.csv', sep=',', header=[0],
        #                  skiprows=[1])
        df.drop('Unnamed: 0', axis=1, inplace=True)
        df['Y'] = df['Y'].replace({'yes': 1, 'no': 0})
        df['X2'] = df['X2'].replace({1: 1, 2: 0})
        # # 3 进行numeric归一化方式的判断，整理出list，传入workflow参数
        # int_cas = []
        age_div_10 = ['X5']
        skip_feat = ['X2', 'X3', 'X4', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11']
        norm_99_99 = ['X12', 'X13', 'X14', 'X15', 'X16', 'X17']
        norm_0_99 = ['X18', 'X19', 'X20', 'X21', 'X22', 'X23']
        norm_1_99 = ['X1']
        df = dchp.work_flow(df, y_labels=Y_features, age_div_10=age_div_10,
                       skip_feat=skip_feat, norm_0_99=norm_0_99,
                       norm_99_99=norm_99_99, norm_1_99=norm_1_99)
        hp.wrt_descrip_txt(df, 'default', Y_feat=Y_features, D_feat=D_features,
                               Y_map=label_maps, D_map=all_protected_attribute_maps,
                               P_map=all_privileged_classes)
        return df

    XD_features = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10'
        , 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20'
        , 'X21', 'X22', 'X23']
    drop = []
    d_features_in_x = ['X2']
    D_features = ['sex']
    X_features = list(set(XD_features) - set(drop) - set(d_features_in_x))
    Y_features = ['Y']

    all_privileged_classes = {'sex': [1]}
    all_protected_attribute_maps = {'X2': {1: 'Male', 0: 'Female'}}
    label_maps = {1: 'yes', 0: 'no'}
    categorical_features = []


    # protected_attribute_names=D_features,
    #         privileged_classes=[all_privileged_classes[x] for x in D_features],
    dataset_default = DefaultCreditDataset(
        label_name=Y_features[0],
        favorable_classes=[0],
        protected_attribute_names=[],
        privileged_classes=[],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features+Y_features+d_features_in_x,
        metadata={'label_maps': [{1: 'default payment', 0: 'non-default payment'}],
                  'protected_attribute_maps': [all_protected_attribute_maps[x] for x in d_features_in_x]},
        custom_preprocessing=custom_preprocessing_new)
    return dataset_default



def load_preproc_data_heart(protected_attributes=None):
    def custom_preprocessing(df):
        # 根据平均数划分界限
        df['age'] = df['age'].apply(lambda x: np.float(x < 54.4))

        # binary label col
        df['Probability'] = df['Probability'].apply(lambda x: np.float(x == 0))
        # df['Probability'] = pd.DataFrame.where(df, df['Probability'] == 0, other=1)
        return df

    XD_features = ['age'
        , 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope'
        , 'ca', 'thal']
    drop = []
    D_features = ['age']
    X_features = list(set(XD_features) - set(drop) - set(D_features))

    Y_features = ['Probability']
    all_privileged_classes = {"age": [1]}
    all_protected_attribute_maps = {'age': {1: 'Old', 0: 'Young'}}
    label_maps = {1: 'absence', 0: 'presence'}
    def custom_preprocessing_new(df):

        r_y_map = {'absence': 1, 'presence': 0}
        # 1 dropna不用特地做
        # 2 敏感属性优劣势处理 标签优劣势处理 特判的列表例如age

        df['Probability'] = df['Probability'].apply(lambda x: int(x == 0))
        # 3 进行numeric归一化方式的判断，整理出list，传入workflow参数
        int_cas = []
        age_div_10 = ['age']
        skip_feat = ['sex', 'cp', 'fbs', 'restecg',
                     'exang', 'slope', 'ca', 'thal']
        norm_0_9 = ['oldpeak']
        norm_1_9 = ['trestbps', 'chol']  # default

        df = hhp.work_flow(df, y_labels=Y_features, age_div_10=age_div_10, skip_feat=skip_feat,
                       norm_0_9=norm_0_9)
        hp.wrt_descrip_txt(df, 'heart', Y_feat=Y_features, D_feat=D_features,
                               Y_map=label_maps, D_map=all_protected_attribute_maps,
                               P_map=all_privileged_classes)
        return df

    categorical_features = []

    # pri classes 年轻的人患病可能性较小 属于优势群体 因此上面的lambda表达式给的也是小于号
    # 同时 prob为0 表示没有患病可能的属于优势


    # protected attr maps

    # protected_attribute_names=D_features,
    #         privileged_classes=[all_privileged_classes[x] for x in D_features],
    dataset_heart = HeartDataset(
        label_name=Y_features[0],
        favorable_classes=[1],
        protected_attribute_names=[],
        privileged_classes=[],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features+Y_features+D_features,
        metadata={'label_maps': [{1.0: 'absence', 0.0: 'presence'}],
                  'protected_attribute_maps': [all_protected_attribute_maps[x] for x in D_features]},
        custom_preprocessing=custom_preprocessing_new)
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

    XD_features = ['school'
        , 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason'
        , 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
                   'higher'
        , 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1'
        , 'G2']
    drop = []
    D_features = ['sex']
    X_features = list(set(XD_features) - set(drop) - set(D_features))
    Y_features = ['Probability']

    all_privileged_classes = {'sex': [1]}
    all_protected_attribute_maps = {'sex': {1: 'Male', 0: 'Female'}}
    label_maps = {1: 'higher than mean', 0: 'lower than mean'}
    categorical_features = []
    def custom_preprocessing_new(df):
        df['sex'] = df['sex'].replace({'F': 0, 'M': 1})
        mean_of_y = df['Probability'].mean()
        df['Probability'] = df['Probability'].apply(lambda x: int(x >= mean_of_y))
        # 3 进行numeric归一化方式的判断，整理出list，传入workflow参数
        int_cas = []
        # age_div_10 = ['age']
        skip_feat = ['sex',
                     'age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
                     'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']

        df = shp.work_flow(df, y_labels=Y_features, skip_feat=skip_feat)
        hp.wrt_descrip_txt(df, 'student', Y_feat=Y_features, D_feat=D_features,
                               Y_map=label_maps, D_map=all_protected_attribute_maps,
                               P_map=all_privileged_classes)
        return df
    #     protected_attribute_names=D_features,
    #         privileged_classes=[all_privileged_classes[x] for x in D_features],
    dataset_student = StudentDataset(
        label_name=Y_features[0],
        favorable_classes=[1],
        protected_attribute_names=[],
        privileged_classes=[],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features+Y_features+D_features,
        metadata={'label_maps': [{1.0: 'higher than mean', 0.0: 'lower than mean'}],
                  'protected_attribute_maps': [all_protected_attribute_maps[x] for x in D_features]},
        custom_preprocessing=custom_preprocessing_new)
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