"""
1. Finds best parameter for SVM
2. Trains model with this parameter
3. Finds top 10 words with highest coeff
"""
import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.svm import SVC


def main():
    newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])

    transformer = TfidfVectorizer()
    tfidf = transformer.fit_transform(newsgroups.data)

    best_param = best_C_for_SVM(tfidf, newsgroups.target)
    print "Best C for SVM:", best_param

    clf = SVC(kernel='linear', random_state=241, C=best_param)
    # clf = SVC(kernel='linear', random_state=241, C=1.0)
    clf.fit(tfidf, newsgroups.target)

    top_10_idx = np.array(clf.coef_.indices)[np.abs(np.array(clf.coef_.data)).argsort()[-10:]]
    top_10_words = np.array(transformer.get_feature_names())[top_10_idx]
    top_10_words.sort()
    print "Top 10 words with highest coef:", ','.join(top_10_words)


def best_C_for_SVM(X, y):
    grid = {'C': np.power(10.0, np.arange(-5, 6))}
    cv = KFold(n_splits=5, shuffle=True, random_state=241)
    clf = SVC(kernel='linear', random_state=241)
    gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
    gs.fit(X, y)

    mean_scores = gs.cv_results_.get('mean_test_score')
    params = gs.cv_results_.get('params')
    best_score = -1000000
    best_param = 0
    for i in range(len(mean_scores)):
        print mean_scores[i]
        if mean_scores[i] > best_score:
            best_score = mean_scores[i]
            best_param = params[i]['C']
    return best_param


if __name__ == "__main__":
    # calling the main function
    main()