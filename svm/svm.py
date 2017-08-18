"""
Finding support vector for SVM
"""
import pandas
from sklearn.svm import SVC

train_data = pandas.read_csv('svm-data.csv', header=None)

y_train = train_data[0]
X_train = train_data.drop(0, inplace=False, axis=1) 

svm = SVC(C=100000, random_state=241)
svm.fit(X_train, y_train)
print 'Indices of support vectors: ', svm.support_