import numpy
import pandas
from scipy.stats.stats import pearsonr

from collections import defaultdict

data = pandas.read_csv('data/train.csv', index_col='PassengerId')
# print data[:10]
# print data.head()
# print data['Pclass'].value_counts()
# print data['Sex'].value_counts()
survived = data['Survived'].value_counts().to_dict()[1]
all_passengers = data['Survived'].size
survived_percent = 100.0 * survived / all_passengers
print "Survived (%) : " + "{0:.2f}".format(survived_percent)

first_class_passengers = data['Pclass'].value_counts().to_dict()[1]
fc_passengers_pc = first_class_passengers * 100.0 / all_passengers
print "First class passengers (%): " + "{0:.2f}".format(fc_passengers_pc)

print data['Age'].mean()
print data['Age'].median()

print pearsonr(data['SibSp'], data['Parch'])
print numpy.corrcoef(data['SibSp'], data['Parch'])

femaleName = defaultdict(int)

# print data['Name']

for name in data['Name']:
    miss = name.split(', Miss. ', 1)
    mrs = name.split(', Mrs. ', 1)

    if len(miss) > 1:
        firstName = miss[1]
        femaleName[firstName] += 1
        # firstName = miss[1].split(' ', 1)
        # femaleName[firstName[0]] += 1
    elif len(mrs) > 1:
        firstName = mrs[1]
        femaleName[firstName] += 1
        # firstName = mrs[1].split(' ', 1)
        # femaleName[firstName[0]] += 1

namesSortedByFreqcy = sorted(femaleName, key=femaleName.get, reverse=True)
print namesSortedByFreqcy[0]