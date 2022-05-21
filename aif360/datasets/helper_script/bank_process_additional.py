import sys
sys.path.append("../")
import numpy as np

"""
Pre-process the raw data of Bank Marketing,
convert the text to numerical data.
"""

# list all the values of enumerate features
job = ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
                                       "blue-collar","self-employed","retired","technician","services"]
marital = ["married","divorced","single","unknown"]
education = ["unknown","basic.9y","high.school","university.degree","professional.course","basic.4y","basic.6y","illiterate"]
default = ["no","yes","unknown"]
housing = ["no","yes","unknown"]
loan = ["no","yes","unknown"]
contact = ["unknown","telephone","cellular"]
month = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
day_of_week = ["mon","thu","wed","tue","fri","wed"]
poutcome = ["nonexistent","other","failure","success"]
output = ["no","yes"]

data = []
with open("../datasets/bank_raw/bank-additional.csv", "r") as ins:
    for line in ins:
        line = line.strip()
        features = line.split(';')
        features[0] = np.clip(int(features[0]) / 10, 1, 9)
        features[1] = job.index(features[1].split('\"')[1])
        features[2] = marital.index(features[2].split('\"')[1])
        features[3] = education.index(features[3].split('\"')[1])
        features[4] = default.index(features[4].split('\"')[1])
        features[5] = housing.index(features[5].split('\"')[1])
        features[6] = loan.index(features[6].split('\"')[1])
        features[7] = contact.index(features[7].split('\"')[1])
        features[8] = month.index(features[8].split('\"')[1])
        features[9] = day_of_week.index(features[9].split('\"')[1])
        features[10] = np.clip(int(features[10]) / 100, 1, 9)
        features[11] = int(features[11])
        if features[12] == 999:
            features[12] = 0
        else:
            features[12] = 1
        features[13] = int(features[13])

        features[14] = poutcome.index(features[14].split('\"')[1])
        if float(features[15])< 0:
            features[15] = int(0)
        else:
            features[15] = int(1)

        if float(features[16])>93:
            features[16] = int(0)
        else:
            features[16] = int(1)

        a=abs(float(features[17]))
        features[17] =  np.clip(int(a) / 10, 1, 9)

        if float(features[18]) > 4.5:
            features[18] = int(0)
        else:
            features[18] = int(1)
        b=abs(float(features[19]))
        features[19] = np.clip(int(b) / 1000, 1, 9)
        features[20] = output.index(features[20].split('\"')[1])
        data.append(features)
data = np.asarray(data)

np.savetxt("../datasets/bank-additional-1", data, fmt="%d",delimiter=",")