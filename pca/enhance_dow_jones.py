import numpy
import pandas
from sklearn.decomposition import PCA


def main():
    X = pandas.read_csv('close_prices.csv')
    X.drop('date', inplace=True, axis=1)
    print "Initial number of features in dataset:", len(X.columns)

    pca = fitted_pca(X, 10)

    min_n_components = min_components(pca, 0.9)
    print "Minimum number of components to explain 90% of data:", min_n_components

    # fitting with minimum components (though it is not necessary, could use PCA with 10)
    pca = fitted_pca(X, min_n_components)

    y = pandas.read_csv('djia_index.csv')
    y.drop('date', inplace=True, axis=1)

    # Apply dimensionality reduction to X, we use min_n_components features, much less than initially
    X_trunced = pca.transform(X)

    first_feature_array = X_trunced[:, 0]
    y_array = y['^DJI'].values
    coer_coeff = numpy.corrcoef(first_feature_array, y_array)[0][1]
    # if |coer_coeff|=1, then there is a linear dependency btw two sets. If 0, then there is no dependency at all.
    print "Correlation between first (reduced) feature and Dow Jones index:", coer_coeff
    print "Correlation is", "good" if (1 - abs(coer_coeff)) < 0.2 else "bad"

    valuable_feature = get_feature_with_max_weight(X, pca.components_[0])
    print "Most weighted feature in first component:", valuable_feature


def get_feature_with_max_weight(X, component_array):
    max_w = -100000
    idx = -1
    for i, weight in enumerate(component_array):
        if weight > max_w:
            max_w = weight
            idx = i

    return X.columns[idx]


def fitted_pca(X, n_components):
    """
    Reduce dataset with n_components as number of features
    :type n_components: number of features to leave
    """
    pca = PCA(n_components=n_components)
    pca.fit(X)
    return pca


def min_components(pca, max_total_ratio):
    """
    Returns -1, if can't find minimum number for given max_total_ratio
    """
    _sum = 0
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        _sum += ratio
        if _sum > max_total_ratio:
            return i + 1

    return -1


if __name__ == "__main__":
    main()
