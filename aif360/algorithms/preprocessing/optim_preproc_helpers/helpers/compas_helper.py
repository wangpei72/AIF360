import math
import sys
import os
import re
import pandas as pd

from aif360.algorithms.preprocessing.optim_preproc_helpers import helper

sys.path.append("../")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.stats import chi2_contingency, kruskal, f_oneway, normaltest, bartlett
import numpy as np
from datetime import date

data_set_list = ['adult', 'compas', 'german', 'bank',
                 'default', 'heart', 'student',
                 'home_credit']

csv_set_list = ['../../../data/raw/adult/adult.csv',
                '../../../data/raw/compas/compas-scores-two-years.csv',
                '../../../data/raw/german/german.csv',
                '../../../data/raw/bank/bank-additional-full.csv',
                '../../../data/raw/default/default_of_credit_card_clients.csv',
                '../../../data/raw/heart/processed.cleveland.data.csv',
                '../../../data/raw/student/Student.csv',
                '../../../data/raw/home/home.csv'
                ]

meps_set_list = ['meps15', 'meps16']
meps_csv_list = ['../../../data/raw/meps/h181.csv',
                 '../../../data/raw/meps/h192.csv']


def map_unique_as_num(df, colum_name):
    list_type = df[colum_name].unique()
    dic = dict.fromkeys(list_type)
    for i in range(len(list_type)):
        dic[list_type[i]] = i
    df[colum_name] = df[colum_name].map(dic)


def print_cols(df):
    cols = df.columns
    for i in cols:
        print(i)


def binary_numeric_by_param(df, col_name, param):
    df[col_name] = df[col_name].apply(lambda x: np.int(x > param))


def binary_numeric_by_mean(df, col_name):
    mean = df[col_name].mean()
    df[col_name] = df[col_name].apply(lambda x: np.int(x >= mean))


def age_normalize(df, col_name):
    df[col_name] = df[col_name].apply(lambda x: int(int(x) / 10))
    df.loc[df[col_name] < 1, col_name] = 1
    df.loc[df[col_name] > 9, col_name] = 9


def clip_numeric_normalize(df, col_name):
    normal_to_a_b(df, col_name, 1, 9)
    df.loc[df[col_name] < 1, col_name] = 1
    df.loc[df[col_name] > 9, col_name] = 9


def get_num_base(x):
    x = abs(x)
    n = int(math.log10(x))
    n_pow = int(math.pow(10, n))
    return n_pow


def normal_to_0_10(df, col_name):
    normal_to_a_b(df, col_name, 0, 10)
    # df[col_name] = df[col_name].apply(lambda x: int(int(x) / get_num_base(x)))
    df.loc[df[col_name] < 1, col_name] = 0
    df.loc[df[col_name] > 9, col_name] = 10


def normal_to_1_99(df, col_name):
    normal_to_a_b(df, col_name, 1, 99)
    df.loc[df[col_name] < 1, col_name] = 1
    df.loc[df[col_name] > 99, col_name] = 99


def normal_to_a_b(df, col_name, a, b):
    #     （1）首先找到样本数据Y的最小值Min及最大值Max
    # （2）计算系数为：k=（b-a)/(Max-Min)
    # （3）得到归一化到[a,b]区间的数据：norY=a+k(Y-Min)
    # 进行计算之前应该将所有包含na会导致计算失败的行drop掉
    df.dropna(subset=[col_name], inplace=True)
    min = df[col_name].min()
    max = df[col_name].max()
    k = np.float(b - a) / np.float(max - min)
    df[col_name] = df[col_name].apply(lambda x: int(a + k * (x - min)))


def age_to_div_10(df, col_name):
    df[col_name] = df[col_name].apply(lambda x: np.int(x / 10))


def normal_to_0_99(df, col_name):
    normal_to_a_b(df, col_name, 0, 99)
    df.loc[df[col_name] < 0, col_name] = 0
    df.loc[df[col_name] > 99, col_name] = 99


def do_c_days_from_compas(df, col_name):
    df.dropna(subset=[col_name], inplace=True)
    df.loc[df[col_name] <= 100, col_name] *= 0.09
    df.loc[df[col_name] > 100, col_name] = 10
    df[col_name] = df[col_name].apply(lambda x: np.int(x))

def do_days_b_screening_arrest(df, col_name):
    #     days_b_screening_arrest
    df.dropna(subset=[col_name], inplace=True)
    df.loc[df[col_name] == 0, col_name] = 0
    df.loc[df[col_name] > 0, col_name] = 1
    df.loc[(df[col_name]<0) & (df[col_name]>-1), col_name] = -1
    df.loc[df[col_name] < -1, col_name] = -2
    df[col_name] = df[col_name].apply(lambda x: np.int(x))


def work_flow(df, y_labels, skip_feat=None, binary_0_feat=None,
              age_feat=None, binary_avg_feat=None, norm_1_99=None,
              norm_0_10=None, norm_0_9=None, age_div_10=None, norm_0_99=None,
              days=None,  c_days_from_compas=None, days_b_screening_arrest=None):
    for i in df.columns:
        if i == 'Unnamed: 0':
            continue
        elif y_labels is not None and i in y_labels:
            continue
        elif skip_feat is not None and i in skip_feat:
            continue
        elif days_b_screening_arrest is not None and i in days_b_screening_arrest:
            do_days_b_screening_arrest(df, i)
        elif c_days_from_compas is not None and i in c_days_from_compas:
            do_c_days_from_compas(df, i)
        elif norm_0_9 is not None and i in norm_0_9:
            normal_to_a_b(df, i, 0, 9)
        elif days is not None and i in days:
            normal_to_0_99(df, i)
        elif age_div_10 is not None and i in age_div_10:
            age_to_div_10(df, i)
        elif norm_0_99 is not None and i in norm_0_99:
            normal_to_0_99(df, i)
        elif norm_1_99 is not None and i in norm_1_99:
            normal_to_1_99(df, i)
        elif norm_0_10 is not None and i in norm_0_10:
            normal_to_0_10(df, i)
        elif binary_0_feat is not None and i in binary_0_feat:
            binary_numeric_by_param(df, i, 0)
        elif binary_avg_feat is not None and i in binary_avg_feat:
            binary_numeric_by_mean(df, i)
        elif age_feat is not None and i in age_feat:
            age_normalize(df, i)
        elif df[i].dtype == np.dtype('int64'):
            clip_numeric_normalize(df, i)
        elif df[i].dtype == np.dtype('float64'):
            clip_numeric_normalize(df, i)
        elif df[i].dtype == np.dtype('O'):
            map_unique_as_num(df, i)
        else:
            print('none supported dtype in df')
    print('work flow done')
    return df

convert = {'Possession of Cocaine': 'Possession of Drug',
           'Battery': 'Battery',
           'Felony DUI (level 3)': 'Felony DUI (level 3)',
           'Criminal Mischief': 'Criminal Mischief',
           'Possession Of Heroin': 'Possession Of drug',
           'Felony Driving While Lic Suspd': 'Felony Driving While License Suspended',
           'Driving While License Revoked': 'Driving While License Revoked',
           'Grand Theft in the 3rd Degree': 'Grand Theft in the third Degree',
           'arrest case no charge': 'arrest case no charge',
           'Possession Of Alprazolam': 'Possession Of Drug',
           'Pos Cannabis W/Intent Sel/Del': 'Possession of Drug',
           'Resist/Obstruct W/O Violence': 'Resist Officer Without Violence',
           'DUI Level 0.15 Or Minor In Veh': 'DUI Level 0.15 Or Minor In Vehicle',
           'Aggravated Assault W/Dead Weap': 'Aggravated Assault With Deadly Weapon',
           'Susp Drivers Lic 1st Offense': 'Operating With Suspended Driving License first Offense',
           'Aggrav Battery w/Deadly Weapon': 'Aggravated Assault With Deadly Weapon',
           'Felony Petit Theft': 'Felony Petit Theft',
           'Tampering With Physical Evidence': 'Tampering With Physical Evidence',
           'Burglary Structure Unoccup': 'Burglary Unoccupied Dwelling',
           'Unlaw LicTag/Sticker Attach': 'Unlawful Sticker Attachment',
           'Offer Agree Secure For Lewd Act': 'Lewdness Violation',
           'Burglary Unoccupied Dwelling': 'Burglary Unoccupied Dwelling',
           'Poss3,4 Methylenedioxymethcath': 'Possession of drug',
           'Driving License Suspended': 'Driving License Suspended',
           'Aggravated Assault w/Firearm': 'Aggravated Assault With Deadly Weapon',
           'False Imprisonment': 'Falsely Imprisonment',
           'Poss Of RX Without RX': 'Possession of Controlled Substance Without Prescription',
           'Defrauding Innkeeper $300/More': 'Defrauding Innkeeper',
           'Ride Tri-Rail Without Paying': 'Ride Railroad Without Paying',
           'Possession Of Methamphetamine': 'Possession Of Drug',
           'Petit Theft $100- $300': 'Petit Theft $100- $300',
           'Aggravated Battery / Pregnant': 'Aggravated Battery with Pregnant',
           'Leaving the Scene of Accident': 'Leaving the Scene of Accident',
           'Aggravated Assault W/dead Weap': 'Aggravated Assault With Deadly Weapon',
           'Resist Officer w/Violence': 'Resist Officer With Violence',
           'Grand Theft (Motor Vehicle)': 'Grand Theft (Motor Vehicle)',
           'Stalking': 'Stalking',
           'Felony Battery (Dom Strang)': 'Felony Battery',
           'Possess Cannabis/20 Grams Or Less': 'Possession of Drug',
           'Driving Under The Influence': 'Driving Under The Influence',
           'Carrying Concealed Firearm': 'Carrying Concealed Firearm',
           'Battery on Law Enforc Officer': 'Battery on Law Enforcement Officer',
           'Deliver Cannabis': 'Delivery Drug',
           'Burglary Conveyance Occupied': 'Burglary Conveyance Occupied',
           'Del Morphine at/near Park': 'Delivery Drug',
           'Leave Acc/Attend Veh/More $50': 'Leaving Accident and Attended Vehicle with More $50',
           'Poss Pyrrolidinovalerophenone': 'Possession of Drug',
           'Viol Injunct Domestic Violence': 'Violation Injunction Domestic Violence',
           'Operating W/O Valid License': 'Operating without Valid License',
           'Disorderly Conduct': 'Resist Officer Without Violence',
           'Battery on a Person Over 65': 'Battery one an elderly',
           'Aggravated Battery': 'Aggravated Battery',
           'Trespass Struct/Conveyance': 'Trespassing Conveyance',
           'Possession of Cannabis': 'Possession of Drug',
           'Burglary Conveyance Assault/Bat': 'Burglary Conveyance Assault or Battery',
           'Felony Battery w/Prior Convict': 'Felony Battery with Prior Convicted',
           'Petit Theft': 'Petit Theft',
           'Burglary Dwelling Occupied': 'Burglary Dwelling Occupied',
           'DUI Property Damage/Injury': 'DUI Property Damage or Injury',
           'Gambling/Gamb Paraphernalia': 'Gambling Drug',
           'Fighting/Baiting Animals': 'Animal Abuse',
           'False Ownership Info/Pawn Item': 'Falsely Ownership Info or Pawn Item',
           'Assault': 'Assault',
           'Manufacture Cannabis': 'Manufacture Drug',
           'Agg Battery Grt/Bod/Harm': 'Aggravated Battery',
           'Poss Of Controlled Substance': 'Possession Of Controlled Substance',
           'Poss of Cocaine W/I/D/S 1000FT Park': 'Selling Drug',
           'Att Burgl Unoccupied Dwel': 'Burglary Unoccupied Dwelling',
           'Del Cannabis For Consideration': 'Possession of Drug',
           'Aggravated Assault': 'Aggravated Assault',
           'Forging Bank Bills/Promis Note': 'Uttering a Forged Instrument',
           'Burglary Structure Assault/Batt': 'Burglary Conveyance Assault or Battery',
           'Opert With Susp DL 2nd Offens': 'Operating With Suspended Driving License second Offense',
           'Robbery W/Firearm': 'Robbery with Deadly Weapon',
           'Cruelty Toward Child': 'Child Abuse',
           'Fleeing or Eluding a LEO': 'Fleeing or Eluding a Law Enforcement Officer',
           'Disorderly Intoxication': 'Disorderly Intoxication',
           'Burglary Conveyance Unoccup': 'Burglary Conveyance Unoccupied',
           'Crim Use of Personal ID Info': 'Criminal Use of Driver License Info',
           'Poss Wep Conv Felon': 'Possession of Weapon with convicted felony',
           'Burglary Dwelling Armed': 'Burglary Dwelling Armed',
           'Possession Firearm School Prop': 'Possession Firearm School Property',
           'Possession of Benzylpiperazine': 'Possession of Drug',
           'Cash Item w/Intent to Defraud': 'Cash Item With Intention to Defrauding',
           'Crimin Mischief Damage $1000+': 'Criminal Mischief Damage $1000+',
           'Unauth Poss ID Card or DL': 'Unauthorized Possession of ID Card or Driving License',
           'Trespass Private Property': 'Trespassing Private Property',
           'Assault Law Enforcement Officer': 'Assault Law Enforcement Officer',
           'Fraudulent Use of Credit Card': 'Fraudulent Use of Credit Card',
           'Littering': 'Littering',
           'Poss Contr Subst W/o Prescript': 'Possession of Controlled Substance Without Prescription',
           'Restraining Order Dating Viol': 'Restraining Order Dating Violence',
           'Possession Burglary Tools': 'Possession of Burglary Tools',
           'Grand Theft of a Fire Extinquisher': 'Grand Theft of a Fire Extinquisher',
           'Fleeing Or Attmp Eluding A Leo': 'Fleeing or Eluding a Law Enforcement Officer',
           'Traffick Amphetamine 28g><200g': 'Trafficking Drug 28g><200g',
           'Agg Fleeing and Eluding': 'Aggravated Fleeing and Eluding',
           'Trespassing/Construction Site': 'Trespassing Construction Site',
           'Reckless Driving': 'Reckless Driving',
           'Agg Abuse Elderlly/Disabled Adult': 'Aggravated Abuse Elderly or disabled Adult',
           'Dealing in Stolen Property': 'Dealing in Stolen Property',
           'Defrauding Innkeeper': 'Defrauding Innkeeper',
           'DUI/Property Damage/Persnl Inj': 'DUI with Property Damage and Person Injury',
           'Grand Theft Firearm': 'Grand Theft with deadly weapon',
           'Kidnapping / Domestic Violence': 'Kidnapping and Domestic Violence',
           'DUI - Enhanced': 'DUI - Enhanced',
           'Failure To Return Hired Vehicle': 'Failure To Return Hired Property',
           'Exposes Culpable Negligence': 'Exposes Culpable Negligence',
           'Opert With Susp DL 2ND Offense': 'Operating With Suspended Driving License second Offense',
           'Use of Anti-Shoplifting Device': 'Possession of Anti-Shoplifting Device',
           'Possession of Hydromorphone': 'Possession of Drug',
           'Uttering a Forged Instrument': 'Uttering a Forged Instrument',
           'Stalking (Aggravated)': 'Aggravated Stalking',
           'Purchase Cannabis': 'Purchase Drug',
           'DWI w/Inj Susp Lic / Habit Off': 'DUI with Injury Suspended Driver License Habit Offense',
           'Lve/Scen/Acc/Veh/Prop/Damage': 'DUI with Property Damage and Person Injury',
           'Child Abuse': 'Child Abuse',
           'Tamper With Witness/Victim/CI': 'Tampering With Witness or Victim or Confidential Informant',
           'Imperson Public Officer or Emplyee': 'Impersonate Public Officer or Employee',
           'Felony Batt(Great Bodily Harm)': 'Aggravated Battery',
           'Possession of Butylone': 'Possession of Drug',
           'Attempted Robbery  No Weapon': 'Attempted Robbery Without Weapon',
           'Felony Battery': 'Felony Battery',
           'Retail Theft $300 2nd Offense': 'Retail Theft $300 second Offense',
           'Burglary Dwelling Assault/Batt': 'Burglary Dwelling Assault or Battery',
           'Prowling/Loitering': 'Prowling and Loitering',
           'Unlawful Conveyance of Fuel': 'Unlawful Conveyance of Fuel',
           'Del Cannabis At/Near Park': 'Delivery Drug',
           'Deliver Cocaine 1000FT Store': 'Delivery Drug',
           'Possession Of 3,4Methylenediox': "Possession of drug",
           'Deliver Cocaine': 'Delivery Drug',
           'Possession Of Amphetamine': 'Possession Of Drug',
           'Fabricating Physical Evidence': 'Fabricating Physical Evidence',
           'Conspiracy to Deliver Cocaine': 'Delivery Drug',
           'Possession of Morphine': 'Possession of Drug',
           'Neglect Child / Bodily Harm': 'Child Abuse',
           'Sex Offender Fail Comply W/Law': 'Sexual Offender Failure to Comply with Law officer',
           'Fail To Redeliver Hire Prop': 'Failure to Return Hired Property',
           'Simulation of Legal Process': 'Simulation of Legal Process',
           'Robbery / Weapon': 'Robbery with Deadly Weapon',
           'Robbery Sudd Snatch No Weapon': 'Robbery',
           'Disrupting School Function': 'Unlawful Disturb Education or Institute',
           'Poss/Sell/Del Cocaine 1000FT Sch': "Selling Drug",
           'Lewd or Lascivious Molestation': 'Lewdness Violation',
           'Poss of Methylethcathinone': 'Possession of drug',
           'Robbery / No Weapon': "Robbery",
           'Grand Theft of the 2nd Degree': 'Grand Theft of the second Degree',
           'Possession Of Buprenorphine': 'Possession Of Drug',
           'Possession of Codeine': 'Possession Of Drug',
           'Criminal Mischief Damage <$200': 'Criminal Mischief Damage <$200',
           'Unlaw Use False Name/Identity': 'Criminal Use of Falsely Identification',
           'Viol Prot Injunc Repeat Viol': "Violation Protect Injunction Repeat Violence",
           'Theft': 'Theft',
           'Viol Pretrial Release Dom Viol': "Domestic Violence",
           'Corrupt Public Servant': 'Corrupt Public officer',
           'Burglary With Assault/battery': 'Burglary With Assault or battery',
           'Harm Public Servant Or Family': 'Harm Public officer Or Family',
           'Aggrav Stalking After Injunctn': 'Aggravated Stalking After Injunction',
           'Possess Drug Paraphernalia': 'Possession of Drug',
           'DUI- Enhanced': 'DUI- Enhanced',
           'Lewdness Violation': 'Lewdness Violation',
           'Possession of Hydrocodone': 'Possession of Drug',
           'Lewd Act Presence Child 16-': 'Lewdness Violation',
           'Poss Cocaine/Intent To Del/Sel': 'Selling Drug',
           'Possession of Oxycodone': 'Possession of Drug',
           'Video Voyeur-<24Y on Child >16': 'Voyeurism',
           'Shoot Into Vehicle': 'Aggravated Assault',
           'Live on Earnings of Prostitute': 'Live on Earnings of Prostitution',
           'Armed Trafficking in Cannabis': 'Armed Trafficking in Drug',
           'Retail Theft $300 1st Offense': 'Retail Theft $300 first Offense',
           'Viol Injunction Protect Dom Vi': 'Violation Injunction Protect Domestic Violence',
           'Fel Drive License Perm Revoke': 'Felony Driving License Permanently Revoked',
           'Possess w/I/Utter Forged Bills': 'Uttering a Forged Instrument',
           'Possession Of Carisoprodol': 'Possession Of Drug',
           'Unlicensed Telemarketing': 'Unlicensed Telemarketing',
           'Posses/Disply Susp/Revk/Frd DL': 'Possession of Suspended Driving License',
           'Att Burgl Struc/Conv Dwel/Occp': 'Attempted Burglary Conveyance Occupied',
           'Possession Of Cocaine': 'Possession Of Drug',
           'Obstruct Fire Equipment': 'Obstruct Fire Equipment',
           'Possession of Alcohol Under 21': 'Possession of Alcohol Under 21',
           'Possession of Methadone': 'Possession of drug',
           'Sale/Del Counterfeit Cont Subs': 'Sale delivery Counterfeit Controlled Substance',
           'Sex Battery Deft 18+/Vict 11-': 'Sexual Battery',
           'DWLS Canceled Disqul 1st Off': 'Driving while license suspended canceled disqualified first offense',
           'Purchase Of Cocaine': 'Purchase Of Drug',
           'Delivery of Heroin': 'Delivery of drug',
           'Crim Attempt/Solicit/Consp': 'Criminal Attempted and Solicitation Conspiracy',
           'Violation Of Boater Safety Id': 'Violation Of Boater Safety Id',
           'Flee/Elude LEO-Agg Flee Unsafe': 'Fleeing or Eluding a Law Enforcement Officer',
           'Poss Oxycodone W/Int/Sell/Del': 'Selling Drug',
           'Grand Theft (motor Vehicle)': 'Grand Theft (motor Vehicle)',
           'Offer Agree Secure/Lewd Act': 'Offer Agree Secure Lewdness Act',
           'Depriv LEO of Protect/Communic': 'Depriving Law Enforcement officer of means of protect or communication',
           'Fraud Obtain Food or Lodging': 'Fraudulent Obtain Food or Lodging',
           'Sex Batt Faml/Cust Vict 12-17Y': 'Sexual Battery family custodial Victim 12-17Y',
           'Intoxicated/Safety Of Another': 'Intoxication and Safety Of Another',
           'DWLS Susp/Cancel Revoked': 'Driving while license suspended canceled revoked',
           'Aide/Abet Prostitution Lewdness': 'Aiding and Abet Prostitution Lewdness',
           'Solic to Commit Battery': 'Solicitation to Commit Battery',
           'Battery On Fire Fighter': 'Battery On Fire Fighter',
           'Shoot In Occupied Dwell': 'Shoot In Occupied Dwelling',
           'Voyeurism': 'Voyeurism',
           'Compulsory Attendance Violation': 'Compulsory Attended Violation',
           'Poss Of 1,4-Butanediol': 'Possession Of Drug',
           'Burgl Dwel/Struct/Convey Armed': 'Burglary conveyance with Deadly Weapon',
           'Poss Pyrrolidinovalerophenone W/I/D/S': 'Possession Pyrrolidinovalerophenone with intention to distribute schedule',
           'Interference with Custody': 'Interference with Custody',
           'Deliver Alprazolam': 'Delivery Drug',
           'Attempted Robbery Firearm': "Attempted Robbery With Deadly Weapon",
           'Trespass Other Struct/Conve': 'Trespassing Other Conveyance',
           'Carjacking with a Firearm': 'Grand Theft with Deadly weapon',
           'False 911 Call': 'Falsely 911 Call',
           'Conspiracy Dealing Stolen Prop': 'Conspiracy Dealing Stolen Property',
           'Failure To Pay Taxi Cab Charge': 'Failure To Paying Taxi Cab Charge',
           'Tresspass in Structure or Conveyance': 'Trespassing in Conveyance',
           'Sexual Performance by a Child': "Child Pornography",
           'Trespass Struct/Convey Occupy': 'Trespassing Conveyance Occupied',
           'Violation of Injunction Order/Stalking/Cyberstalking': 'Violation of Injunction Order or Stalking or Cyberstalking',
           'Sell/Man/Del Pos/w/int Heroin': 'selling manufacture delivery possession with intention drug',
           'Exhibition Weapon School Prop': 'Exhibition Weapon School Property',
           'Prostitution/Lewdness/Assign': 'Prostitution or Lewdness or Assignation',
           'Solicit To Deliver Cocaine': 'Solicitation To Delivery Drug',
           'Escape': 'Escape',
           'Carjacking w/o Deadly Weapon': 'Carjacking Without Deadly Weapon',
           'Aggravated Battery On 65/Older': 'Aggravated Battery On Elderly',
           'Crim Attempt/Solic/Consp': 'Criminal Attempted, Solicitation Conspiracy',
           'Traffic Counterfeit Cred Cards': 'Trafficking Counterfeit Credit Card',
           'Trans/Harm/Material to a Minor': 'Transmission material harmful to a minor',
           'Giving False Crime Report': 'Contradict Statement',
           'Abuse Without Great Harm': 'Abuse Without Great Harm',
           'Poss Alprazolam W/int Sell/Del': 'Selling Drug',
           'Structuring Transactions': 'Structuring Transactions',
           'Purchase/P/W/Int Cannabis': 'purchase with intention Drug',
           'False Name By Person Arrest': 'Contradict Statement',
           'Unauth C/P/S Sounds>1000/Audio': 'Unauthorized C/P/S',
           'Possession Of Clonazepam': 'Possession Of Drug',
           'False Info LEO During Invest': 'Falsely Info law enforcement officer During Investigation',
           'Money Launder 100K or More Dols': 'Money Launder 100K or More Dollars',
           'Carry Open/Uncov Bev In Pub': 'Carrying Open Uncovered Beverage In Public',
           'Possession Of Anabolic Steroid': 'Possession Of Anabolic Steroid',
           'Crim Use Of Personal Id Info': 'Criminal Use Of Driver License Info',
           'Criminal Attempt 3rd Deg Felon': 'Criminal Attempted third Degree felony',
           'Sell Cannabis': 'Selling Drug',
           'Possession of Cannabis': 'Possession of Drug',
           'Unlaw Lic Use/Disply Of Others': 'Unlawful Driver License Use and Display Of Others',
           'Use Computer for Child Exploit': 'Child Pornography',
           'Attempted Deliv Control Subst': 'Attempted and Delivery Controlled Substance',
           'Tampering with a Victim': 'Tampering with a Victim',
           'Obstruct Officer W/Violence': 'Resist Officer with Violence',
           'Consume Alcoholic Bev Pub': 'Consume Alcohol Beverage Public',
           'Sexual Battery / Vict 12 Yrs +': 'Sexual Battery with Victim 12Y',
           'Possession Of Fentanyl': 'Possession Of Drug',
           'Del 3,4 Methylenedioxymethcath': 'Delivery drug',
           'Unlawful Use Of Police Badges': 'Unlawful Use Of Police Badges',
           'Battery Spouse Or Girlfriend': 'Battery Spouse Or Girlfriend',
           'Deliver Cocaine 1000FT School': 'Delivery Drug School',
           'False Bomb Report': 'Falsely Bomb Report',
           'Computer Pornography': 'Computer Pornography',
           'Use Of 2 Way Device To Fac Fel': 'Use Of 2 Way Device To Face Felony',
           'Possession Of Lorazepam': 'Possession Of Drug',
           'Robbery W/Deadly Weapon': 'Robbery With Deadly Weapon',
           'Attempt Armed Burglary Dwell': 'Attempted Armed Burglary Dwelling',
           'Insurance Fraud': 'Insurance Fraudulent',
           'Possess Mot Veh W/Alt Vin #': 'Possession of Motor Vehicle With Alternative Number',
           'Delivery of 5-Fluoro PB-22': 'Delivery of Drug',
           'Deliver Cocaine 1000FT Park': 'Delivery Drug',
           'Arson II (Vehicle)': 'Arson (Vehicle)',
           'Possession Of Paraphernalia': 'Possession Of Drug',
           'Contradict Statement': 'Contradict Statement',
           'Consp Traff Oxycodone 28g><30k': 'Conspire Trafficking Drug',
           'Possession of XLR11': 'Possession of XLR11',
           'Unauthorized Interf w/Railroad': 'Unauthorized Interference with Railroad',
           'Counterfeit Lic Plates/Sticker': 'Counterfeit License Plates or Sticker',
           'Possess Cannabis 1000FTSch': 'Possession of Drug',
           'Aggress/Panhandle/Beg/Solict': 'Aggressive Panhandling',
           'Poss Trifluoromethylphenylpipe': 'Possession of Drug',
           'Murder In 2nd Degree W/firearm': 'Murder In Second Degree With Firearm',
           'Traffick Hydrocodone   4g><14g': 'Trafficking Drug',
           'Principal In The First Degree': 'Principal In The first Degree',
           'Deliver Cocaine 1000FT Church': 'Delivery Drug',
           'Att Burgl Conv Occp': 'Attempted Burglary Conveyance Occupied',
           'Unl/Disturb Education/Instui': 'Unlawful Disturb Education or Institute',
           'Possession Of Diazepam': 'Possession Of Diazepam',
           'Interfere W/Traf Cont Dev RR': 'Interference With Trafficking Controlled Device railroad',
           'Deliver Cannabis 1000FTSch': 'Delivery Drug School',
           'Possession of LSD': 'Possession of LSD',
           'Lewd/Lasciv Molest Elder Persn': 'Lewdness Violation',
           'Trespass On School Grounds': 'Trespassing On School Grounds',
           'Throw In Occupied Dwell': 'Throw In Occupied Dwelling',
           'Fail Sex Offend Report Bylaw': 'Failure Sexual Offend Report By Law',
           'Manslaughter W/Weapon/Firearm': 'Manslaughter With Deadly Weapon',
           'Throw Missile Into Pub/Priv Dw': 'Throw Missile Into Public or Private Driveway',
           'Present Proof of Invalid Insur': 'Present Proof of Invalid Insurance',
           'Theft/To Deprive': 'Theft',
           'Poss Unlaw Issue Driver Licenc': 'Possession of Unlawful Issuing Driver License',
           'Extradition/Defendants': 'Extradition Defendants',
           'Tamper With Victim': 'Tampering With Victim',
           'Neglect Child / No Bodily Harm': 'Child Abuse',
           'Poss 3,4 MDMA (Ecstasy)': 'Possession of drug',
           'Violation License Restrictions': 'Violation License Restrictions',
           'Criminal Mischief>$200<$1000': 'Criminal Mischief>$200<$1000',
           'Felon in Pos of Firearm or Amm': 'Possession of Deadly Weapon',
           'Culpable Negligence': 'Culpable Negligence',
           'Uttering Forged Bills': 'Uttering Forged Bills',
           'Possess Countrfeit Credit Card': 'Possession of Counterfeit Credit Card',
           'Leaving Acc/Unattended Veh': 'Leaving Accident with Unattended Vehicle',
           'Hiring with Intent to Defraud': 'Defrauding Hired',
           'Sell or Offer for Sale Counterfeit Goods': 'Selling Counterfeit Goods',
           'Refuse to Supply DNA Sample': 'Refuse to Supply DNA Sample',
           'Felony/Driving Under Influence': 'Felony and Driving Under Influence',
           'DUI Blood Alcohol Above 0.20': 'DUI Blood Alcohol Above 0.20',
           'Aggravated Battery (Firearm/Actual Possession)': 'Aggravated Battery',
           'Exploit Elderly Person 20-100K': 'Exploit Elderly Person 20-100K',
           'Soliciting For Prostitution': 'Solicitation For Prostitution',
           'Battery Emergency Care Provide': 'Battery Emergency Care Provide',
           'Attempted Burg/struct/unocc': 'Attempted Burglary Conveyance unoccupied',
           'Drivg While Lic Suspd/Revk/Can': 'Driving while driver license Suspended',
           'Aggravated Assault W/o Firearm': 'Aggravated Assault Without Weapon',
           'Fail To Obey Police Officer': 'Resist Officer Without Violence',
           'Poss F/Arm Delinq': 'Possession of Firearm Delinquency',
           'Open Carrying Of Weapon': 'Open Carrying Of Weapon',
           'Aggrav Child Abuse-Agg Battery': 'Aggravated Child Abuse',
           'D.U.I. Serious Bodily Injury': 'D.U.I. Serious Bodily Injury',
           'Strong Armed  Robbery': 'Strong Armed  Robbery',
           'Accessory After the Fact': 'Accessory After the Fact',
           'Burglary Assault/Battery Armed': 'Burglary or Assault or Battery with Deadly Weapon',
           'Deliver 3,4 Methylenediox': 'Delivery drug',
           'Att Tamper w/Physical Evidence': 'Attempted Tampering with Physical Evidence',
           'Lewd/Lasc Battery Pers 12+/<16': 'Lewdness violation',
           'Expired DL More Than 6 Months': "Operating with invalid Driver license",
           'Discharge Firearm From Vehicle': 'shoot Firearm From Vehicle',
           'Solicitation On Felony 3 Deg': 'Solicitation On Felony third Degree',
           'Poss/Sell/Deliver Clonazepam': 'Selling Clonazepam',
           'Traffick Oxycodone     4g><14g': 'Trafficking Drug 4g><14g',
           'Fail Register Vehicle': 'Failure Register Vehicle',
           'Grand Theft Dwell Property': 'Grand Theft Dwelling Property',
           'Felony Committing Prostitution': 'Prostitution',
           'Prostitution/Lewd Act Assignation': 'Prostitution/Lewdness Act Assignation',
           'Solicit Deliver Cocaine': 'Delivery Drug',
           'Poss of Vessel w/Altered ID NO': 'Possession of Vessel with Alternative ID number',
           'Threat Public Servant': 'Threat Public officer',
           'Poss/Sell/Del/Man Amobarbital': 'Selling Drug',
           'Use Scanning Device to Defraud': 'Use Scanning Device to Defrauding',
           'Poss Drugs W/O A Prescription': 'Possession of Controlled Substance Without Prescription',
           'False Motor Veh Insurance Card': 'Falsely Motor Vehicle Insurance Card',
           'Poss Meth/Diox/Meth/Amp (MDMA)': 'Possession of drug',
           'Burglary Conveyance Armed': 'Burglary Conveyance with Deadly Weapon',
           'Aiding Escape': 'Aiding Escape',
           'PL/Unlaw Use Credit Card': 'Unlawful Use Credit Card',
           'Carrying A Concealed Weapon': 'Carrying A concealed weapon',
           'Introduce Contraband Into Jail': 'Introduce Contraband Into Jail',
           'Lease For Purpose Trafficking': 'Lease For Purpose Trafficking',
           'Grand Theft in the 1st Degree': 'Grand Theft in the first Degree',
           'Grand Theft on 65 Yr or Older': 'Grand Theft on Elderly',
           'Trespass Structure w/Dang Weap': 'Trespassing Conveyance with Deadly Weapon',
           'Murder in 2nd Degree': 'Murder in second Degree',
           'Poss Anti-Shoplifting Device': 'Possession of Anti-Shoplifting Device',
           'Attempt Burglary (Struct)': 'Attempted Burglary ',
           'Attempted Robbery  Weapon': 'Attempted Robbery with Weapon',
           'Agg Assault Law Enforc Officer': 'Aggravated Assault Law Enforcement Officer',
           'Tamper With Witness': 'Tampering With Witness',
           'Aggravated Battery (Firearm)': 'Aggravated Battery with Deadly Weapon',
           'Traff In Cocaine <400g>150 Kil': 'Trafficking In Drug',
           'Tresspass Struct/Conveyance': 'Trespassing Conveyance',
           'Poss Firearm W/Altered ID#': 'Possession of Firearm with Alternative ID number',
           'Throw Deadly Missile Into Veh': 'Throw Deadly Missile Into Vehicle',
           'Poss Unlaw Issue Id': 'Possession of Unlawful Issuing Driver License',
           'Fail To Redeliv Hire/Leas Prop': 'Failure to return hired property',
           'Cruelty to Animals': 'Animal abuse',
           'nan': 'Unknown',
           'Misuse Of 911 Or E911 System': "Falsely 911 call",
           'Crlty Twrd Child Urge Oth Act': "Child Abuse",
           'Possession of Ethylone': 'Possession of Drug',
           'Attempted Burg/Convey/Unocc': 'Attempted Burglary Conveyance Unoccupied',
           'Poss of Firearm by Convic Felo': 'Possession of Firearm by Convicted Felony',
           'Obtain Control Substance By Fraud': 'Obtain Controlled Substance By Fraudulent',
           'Fail Sex Offend Report Bylaw': 'Failure Sexual Offender Report By law',
           'DOC/Cause Public Danger': 'DOC and Cause Public Dangerous',
           'Contribute Delinquency Of A Minor': 'Contribute Delinquency Of A Minor',
           'Trespass Structure/Conveyance': 'Trespassing Conveyance',
           'Poss Counterfeit Payment Inst': 'Possession of Counterfeit Paying Instrument',
           'Poss Cntrft Contr Sub w/Intent': 'Possession of Counterfeit Controlled Substance',
           'Pos Methylenedioxymethcath W/I/D/S': 'Selling drug',
           'Poss Tetrahydrocannabinols': "Possession of Drug",
           'License Suspended Revoked': 'License Suspended Revoked',
           'Battery On A Person Over 65': 'Battery On A Person Over 65',
           'Trespass Property w/Dang Weap': 'Trespassing Property With Dangerous Weapon',
           'Consp Traff Oxycodone  4g><14g': 'Trafficking Drug  4g><14g',
           'Agg Fleeing/Eluding High Speed': 'Aggravated Fleeing or Eluding High Speed',
           'Aggr Child Abuse-Torture,Punish': 'Aggravated Child Abuse',
           'Bribery Athletic Contests': 'Bribery Athletic Contests',
           'Purchasing Of Alprazolam': 'Purchase Drug',
           'Del of JWH-250 2-Methox 1-Pentyl': 'Delivery Drug',
           'Dealing In Stolen Property': 'Dealing In Stolen Property',
           'Pos Cannabis For Consideration': 'Possession of Drug',
           'Sel Etc/Pos/w/Int Contrft Schd': 'Selling controlled substance',
           'Murder in the First Degree': 'Murder in the First Degree',
           'Alcoholic Beverage Violation-FL': 'Alcohol Beverage Violation',
           'Uttering Worthless Check +$150': 'Uttering Worthless Check +$150',
           'Burglary Structure Occupied': 'Burglary Conveyance Occupied',
           'Battery On Parking Enfor Speci': 'Battery',
           'Refuse Submit Blood/Breath Test': 'Refuse Submit Blood/Breath Test',
           'Oper Motorcycle W/O Valid DL': 'Operating Motor Without Valid Driver License',
           'Possession Of Phentermine': 'Possession Of Drug',
           'Possession Child Pornography': 'Child Pornography',
           'DUI - Property Damage/Personal Injury': 'DUI - Property Damage and Person Injury',
           'Unemployment Compensatn Fraud': 'Unemployment Compensation Fraudulent',
           'Felony DUI - Enhanced': 'DUI - Enhanced',
           'Fail Obey Driv Lic Restrictions': 'Failure to Obey Driver License Restrictions',
           'Issuing a Worthless Draft': 'Issuing a Worthless Draft',
           'Sel/Pur/Mfr/Del Control Substa': 'Selling Controlled substance',
           'Grand Theft In The 3Rd Degree': 'Grand Theft In The third Degree',
           'Harass Witness/Victm/Informnt': 'Harass Witness or Victim of Confidential Informant',
           'Uttering Forged Credit Card': 'Uttering Forged Credit Card',
           'Leave Accd/Attend Veh/Less $50': 'Leaving accident with attended Vehicle less than $50',
           'Lewd/Lasc Exhib Presence <16yr': 'Lewdness Violation',
           'Poss Pyrrolidinobutiophenone': 'Possession of drug',
           'Offn Against Intellectual Prop': 'Offense against Intellectual Property',
           'Manage Busn W/O City Occup Lic': 'Manage Business without city occupation license',
           'Arson in the First Degree': 'Arson first degree',
           'Sound Articles Over 100': 'Sound Articles Over 100',
           'Falsely Impersonating Officer': 'Falsely Impersonate Officer',
           'Poss Similitude of Drivers Lic': 'Possession of unlawful Driver License',
           'Sale/Del Cannabis At/Near Scho': 'Selling Drug',
           'Poss/pur/sell/deliver Cocaine': 'Selling Drug',
           'Fail Register Career Offender': 'Failure Register Career Offender',
           'Sell Conterfeit Cont Substance': 'Selling Counterfeit Controlled Substance',
           'Aggrav Child Abuse-Causes Harm': 'Aggravated Child Abuse',
           'Possess Controlled Substance': 'Possession of Controlled Substance',
           'Neglect/Abuse Elderly Person': "Elderly Abuse",
           'Fail To Secure Load': 'Failure To Secure Load',
           'Compulsory Sch Attnd Violation': 'Compulsory School Attended Violation',
           'Solicit Purchase Cocaine': 'Solicitation Purchase Drug',
           'Possess Weapon On School Prop': 'Possession Weapon On School Property',
           'Possess/Use Weapon 1 Deg Felon': 'Possession of Weapon first Degree Felony',
           'Cause Anoth Phone Ring Repeat': 'Cause Another Phone Ring Repeat',
           'Prostitution': 'Prostitution',
           'Possess Tobacco Product Under 18': 'Possession of Tobacco Product Under 18',
           'Agg Assault W/int Com Fel Dome': 'Aggravated Assault With intention to commit Domestic Violence'}

def do_description(full_df):
    full_df_is_recid = full_df.drop("two_year_recid", axis=1)
    full_df_two_year_recid = full_df.drop("is_recid", axis=1)
    dict_desc = []
    convert_values = list(convert.values())
    for key, item in convert.items():
        cv = CountVectorizer(stop_words='english')
        item1 = re.sub('[^a-zA-Z]', ' ', item)
        tmp = item1.lower().split()
        matrix = cv.fit_transform(tmp)
        vocab = list(cv.vocabulary_.keys())
        convert[key] = vocab
    # update the charge description columns to new converted list
    full_df_is_recid['c_charge_desc'] = full_df_is_recid['c_charge_desc'].map(convert)
    full_df_is_recid = full_df_is_recid.dropna(subset=['c_charge_desc'])

    full_df_two_year_recid['c_charge_desc'] = full_df_two_year_recid['c_charge_desc'].map(convert)
    full_df_two_year_recid = full_df_two_year_recid.dropna(subset=['c_charge_desc'])

    # TWO_YEAR_RECID
    # Apply one hot encoding for the list of charge description
    mlb = MultiLabelBinarizer(sparse_output=True)

    charge_label = full_df_two_year_recid["two_year_recid"].to_frame()
    charge_label = charge_label.join(
        pd.DataFrame.sparse.from_spmatrix(
            mlb.fit_transform(full_df_two_year_recid.pop('c_charge_desc')),
            index=charge_label.index,
            columns=mlb.classes_))
    SIG = 0.05
    MOD_SIG = 0.1
    cols = charge_label.columns.to_list()
    LABEL = "two_year_recid"
    cols.remove(LABEL)
    feat_keep = []
    feat_maybe = []
    for col in cols:
        contingency = pd.crosstab(charge_label[LABEL], charge_label[col])

        c, p, dof, expected = chi2_contingency(contingency)
        if p < SIG:
            feat_keep.append(col)
            print(col, "and label are not independent - keep, p =", p)
        elif p < MOD_SIG:
            feat_maybe.append(col)
            print(col, "and label may have some relationship - maybe keep, p =", p)
        else:
            print(col, "and label are independent - drop, p =", p)
    list_corr = full_df_two_year_recid.corr(method="spearman")['two_year_recid']
    print("list corr is : ", list_corr)
    two_year_feat = full_df_two_year_recid.columns.to_list()
    print("len feat_keep is ", len(feat_keep))
    print(feat_keep)
    print("len feat_maybe is ", len(feat_maybe))
    print(feat_maybe)
    # feat_keep = feat_keep + feat_maybe + ['two_year_recid']
    feat_keep = feat_keep + ['two_year_recid']
    print(feat_keep)
    charge_label = charge_label[feat_keep]
    for i in charge_label.columns.to_list():
        full_df_two_year_recid[i] = charge_label[i]

    print(full_df_two_year_recid.shape)
    LABEL = "two_year_recid"
    df_fail_two_year = full_df_two_year_recid[full_df_two_year_recid[LABEL] == 0]
    df_pass_two_year = full_df_two_year_recid[full_df_two_year_recid[LABEL] == 1]

    # SIG = 0.05
    # MOD_SIG = 0.1
    # for col in two_year_feat:
    #     pop1 = df_fail_two_year[col]
    #     pop2 = df_pass_two_year[col]
    #     stat1, p1 = normaltest(pop1)
    #     stat2, p2 = normaltest(pop2)
    #     if p1 > SIG and p2 > SIG:
    #         stat, p = bartlett(pop1, pop2)
    #         if p > SIG:
    #             print(col, "meets ANOVA assumptions")
    #         else:
    #             print(col, "--> Kruskal-Wallis, variance is unequal:", p)
    #     else:
    #         print(col, "--> Kruskal-Wallis, not normally distributed:", p1, p2)

    # for col in two_year_feat:
    #     pop1 = df_fail_two_year[col]
    #     pop2 = df_pass_two_year[col]
    #     stat, p = kruskal(pop1, pop2)
    #     if p <= SIG:
    #         print(col, "and label are not independent - keep, p =", p)
    #     elif p <= MOD_SIG:
    #         print(col, "and label may have some relationship - maybe keep, p =", p)
    #     else:
    #         print(col, "and label are independent - drop, p =", p)
    # new_two_year_feat = []
    # for col in two_year_feat:
    #     pop1 = df_fail_two_year[col]
    #     pop2 = df_pass_two_year[col]
    #     stat, p = f_oneway(pop1, pop2)
    #     if p <= SIG:
    #         new_two_year_feat.append(col)
    #         print(col, "and label are not independent - keep, p =", p)
    #     elif p <= MOD_SIG:
    #         print(col, "and label may have some relationship - maybe keep, p =", p)
    #     else:
    #         print(col, "and label are independent - drop, p =", p)

    print(len(two_year_feat))
    print(two_year_feat)
    return full_df_two_year_recid[two_year_feat]
    # new_two_year_feat.remove("two_year_recid")
    # df_1_x = full_df_two_year_recid[new_two_year_feat]
    # df_1_y = full_df_two_year_recid[LABEL]

if __name__ == '__main__':
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
            'v_screening_date', 'v_type_of_assessment', 'type_of_assessment','priors_count.1','decile_score.1']
    X_feat = list(set(XD_features) - set(drop))
    D_features = ['sex', 'race']
    Y_features = ['two_year_recid']  # 优势为label ： 0
    all_privileged_classes = {"sex": [1],
                              "race": [1]}
    # protected attribute maps
    all_protected_attribute_maps = {"sex": {0: 'Male', 1: 'Female'},
                                    "race": {1: 'Caucasian', 0: 'Not Caucasian'}}
    label_maps = {1.0: 'Did recid.', 0.0: 'No recid.'}  # 优势label是 no-recid 无再犯
    features_to_drop = ['compas_screening_date' ]

    df = pd.read_csv('../../../data/raw/compas/compas-scores-two-years.csv')
    df.drop(['id', 'name', 'first', 'last', 'compas_screening_date', 'r_case_number', 'dob',
                  'age_cat', 'c_case_number', 'c_offense_date', 'c_arrest_date', 'r_charge_degree',
                   'r_case_number', 'r_days_from_arrest',
                  'r_offense_date', 'violent_recid', 'vr_case_number',
                  'vr_charge_degree', 'vr_offense_date', 'vr_charge_desc', 'screening_date',
                  'v_screening_date', 'v_type_of_assessment', 'type_of_assessment',
                  'priors_count.1','decile_score.1' ], axis=1, inplace=True)
    # df = df.dropna()
    # 进行日期的计算之前将所有日期都是na的行drop掉
    df.dropna(subset =['out_custody', 'in_custody',
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

    df = work_flow(df, y_labels=Y_features, skip_feat=skip_feat, age_div_10=age_div_10,
                   norm_0_99=norm_0_99, norm_0_9=norm_0_9, days=newly_added_days_feat, c_days_from_compas=c_days,
                   days_b_screening_arrest=days_b)
    df.to_csv('df_compas.csv')
    helper.wrt_descrip_txt(df, 'compas', Y_feat=Y_features, D_feat=D_features, Y_map=label_maps
                           , D_map=all_protected_attribute_maps, P_map=all_privileged_classes)
    print('done')
