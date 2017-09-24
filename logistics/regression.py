"""
1. Use gradient descent to calculate logistic regression.
2. Use L2 regularization
"""
import numpy
import pandas
from math import exp
from sklearn import metrics

EPSILON = 0.00001
MAX_ITERS = 10000


def logicstic_regression(X, y, w, k, C):
    res_1 = 0
    res_2 = 0
    for row in X.itertuples():
        i = row[0]
        x1 = row[1]
        x2 = row[2]
        y_i = y.get_value(i, 0)
        v = y_i * (1 - 1 / (1 + exp(-y_i * (w[0] * x1 + w[1] * x2))))
        res_1 += x1 * v
        res_2 += x2 * v

    l = len(y)
    res_1 = w[0] + k * (res_1 / l - C * w[0])
    res_2 = w[1] + k * (res_2 / l - C * w[1])

    return [res_1, res_2]


def gradient_descent(w_1_0, w_2_0, k, C, X, y):
    w = [w_1_0, w_2_0]
    w_prev = [w_1_0 + 1, w_2_0 + 1]

    i = 0
    while (i < MAX_ITERS) and euclid_distance(w_prev, w) > EPSILON:
        w_prev = w
        w = logicstic_regression(X, y, w, k, C)
        i += 1

    return w


def euclid_distance(w_prev, w):
    diff = [j - i for i, j in zip(w_prev, w)]
    return numpy.linalg.norm(diff)


# probability estimation using sigmoid function
def prob_estimate(X, w1, w2):
    y_score = []
    for e in X:
        y_score.append(1 / (1 + exp(-w1 * e[0] - w2 * e[1])))
    return y_score


def main():
    train_data = pandas.read_csv('data-logistic.csv', header=None)
    y_train = train_data[0]
    X_train = train_data.drop(0, inplace=False, axis=1)

    # calc w for non-regularized function
    w_non_reg = gradient_descent(0, 0, 0.1, 0, X_train, y_train)

    # calc w for L2 regularized function
    w_L2 = gradient_descent(0, 0, 0.1, 10, X_train, y_train)

    matrix = X_train.as_matrix()
    y_scores = prob_estimate(matrix, w_non_reg[0], w_non_reg[1])

    # AUC-ROC metrics is used for binary classifiers
    print "AUC-ROC quality metrics for normal logistic regression", metrics.roc_auc_score(y_train, y_scores)

    y_scores = prob_estimate(matrix, w_L2[0], w_L2[1])
    print "AUC-ROC quality metrics for L2 logistic regression", metrics.roc_auc_score(y_train, y_scores)

if __name__ == "__main__":
    # calling the main function
    main()
