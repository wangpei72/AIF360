import sys

sys.path.append("../")
import numpy as np
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('../../../data/raw/meps/h181.csv', sep=',', header=[0])
    print((df['TOTTCH15'] == 2014).sum())
    print((df['BEGRFY53'] == 2015).sum())
    print(df['INS53X'].mean())
