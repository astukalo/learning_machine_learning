import pandas
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
from collections import namedtuple

def optimalK(data, y_train, k_fold):
    max_score = 0
    best_neighbors = 0
    Klassifier = namedtuple('Klassifier', ['bestK', 'max_score'])
    for i in range(1, 51):
        score = cross_val_score(KNeighborsClassifier(n_neighbors=i), X=data, y=y_train, cv=k_fold, scoring='accuracy',
                                verbose=0)
        mean = score.mean()
        # bigger is better
        if mean > max_score:
            max_score = mean
            best_neighbors = i
    return Klassifier(max_score, best_neighbors)


def main():
    data = pandas.read_csv('wine.data', names=['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash',
                                               'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
                                               'Proanthocyanins', 'Color intensity', 'Hue',
                                               'OD280/OD315 of diluted wines', 'Proline'], header=None)
    y_train = data['Class']
    data.drop('Class', inplace=True, axis=1)

    k_fold = KFold(n_splits=5, random_state=42, shuffle=True)

    # finding optimal K for raw data
    klassifier = optimalK(data, y_train, k_fold)
    print 'Optimal quality:', klassifier.max_score
    print 'Best K for kNN:', klassifier.bestK

    # finding optimal K for scaled data
    data_scaled = scale(X=data)
    klassifier = optimalK(data_scaled, y_train, k_fold)
    print 'Optimal quality (scaled data):', klassifier.max_score
    print 'Best K for kNN (scaled data):', klassifier.bestK

main()