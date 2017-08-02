"""
1. Using Perceptron as simple linear model
2. Comparing accuracy of a model for scaled and non-scaled data
"""
import pandas
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def perceptron_accuracy(X_train, y_train, X_test, y_test):
    clf = Perceptron(random_state=241)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    acc_score = accuracy_score(y_test, y_predict)
    return acc_score

def main():
    train_data = pandas.read_csv('perceptron-train.csv', header=None)
    test_data = pandas.read_csv('perceptron-test.csv', header=None)

    y_train = train_data[0]
    X_train = train_data.drop(0, inplace=False, axis=1) 

    y_test = test_data[0]
    X_test = test_data.drop(0, inplace=False, axis=1) 

    # calculating Perceptron accuracy score on raw datasets
    acc_score = perceptron_accuracy(X_train, y_train, X_test, y_test)
    print 'Accuracy score for non-scaled data:', acc_score

    # standardizing features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # calculating Perceptron accuracy score on standardized datasets
    acc_score_scaled = perceptron_accuracy(X_train_scaled, y_train, X_test_scaled, y_test)
    print 'Accuracy score for scaled data:', acc_score_scaled

    print 'Diff in accuracy score:', acc_score_scaled - acc_score

main()