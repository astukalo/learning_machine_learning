import pandas
from sklearn import metrics

_RECALL_THRESHOLD = 0.7


def main():
    # calc metrics for algos, which return class
    _classified_algos()

    # calc metrics for algos, which return probability
    _prob_classifiers_algos()


def _prob_classifiers_algos():
    scores_data = pandas.read_csv('scores.csv')
    true_data = scores_data['true']
    score_logreg = metrics.roc_auc_score(true_data, scores_data['score_logreg'])
    print 'logreg', score_logreg
    score_svm = metrics.roc_auc_score(true_data, scores_data['score_svm'])
    print 'svm', score_svm
    score_knn = metrics.roc_auc_score(true_data, scores_data['score_knn'])
    print 'knn', score_knn
    score_tree = metrics.roc_auc_score(true_data, scores_data['score_tree'])
    print 'tree', score_tree
    print

    max_prec_logreg = _max_precision(true_data, scores_data['score_logreg'])
    print 'logreg', max_prec_logreg
    max_prec_svm = _max_precision(true_data, scores_data['score_svm'])
    print 'svm', max_prec_svm
    max_prec_knn = _max_precision(true_data, scores_data['score_knn'])
    print 'knn', max_prec_knn
    max_prec_tree = _max_precision(true_data, scores_data['score_tree'])
    print 'tree', max_prec_tree


def _max_precision(true_data, pred_data):
    precision, recall, thresholds = metrics.precision_recall_curve(true_data, pred_data)
    max_precision = 0
    idx = 0
    for r in recall:
        if r >= _RECALL_THRESHOLD:
            if precision[idx] > max_precision:
                max_precision = precision[idx]
        idx += 1
    return max_precision


def _classified_algos():
    F_N, F_P, T_N, T_P = _stats('classification.csv')
    print 'T_P, F_P, F_N, T_N:', T_P, F_P, F_N, T_N

    test_data = pandas.read_csv('classification.csv')
    true_data = test_data['true']
    predicted_data = test_data['pred']
    accuracy_score = metrics.accuracy_score(true_data, predicted_data)
    accuracy = 1.0 * (T_P + T_N) / (T_P + T_N + F_N + F_P)
    print 'accuracy', accuracy_score, '(', accuracy, ')'

    precision_score = metrics.precision_score(true_data, predicted_data)
    precision = (1.0 * T_P) / (T_P + F_P)
    print 'precision', precision_score, '(', precision, ')'

    recall_score = metrics.recall_score(true_data, predicted_data)
    recall = (1.0 * T_P) / (T_P + F_N)
    print 'recall', recall_score, '(', recall, ')'

    f__score = metrics.f1_score(true_data, predicted_data)
    f_score = (2.0 * precision * recall) / (precision + recall)
    print 'recall', f__score, '(', f_score, ')'


def _stats(csv):
    T_P = 0
    F_P = 0
    T_N = 0
    F_N = 0
    with open(csv) as f:
        for line in f:
            vals = line.strip().split(',')
            if vals[1] == '1':
                if vals[0] == '1':
                    T_P += 1
                else:
                    F_P += 1
            elif vals[1] == '0':
                if vals[0] == '0':
                    T_N += 1
                else:
                    F_N += 1
    return F_N, F_P, T_N, T_P


if __name__ == "__main__":
    # calling the main function
    main()