"""This scripts finds the best p for minkowski metrics"""

from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from numpy import linspace
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

boston_data_set = load_boston()  # using dataset with Boston's housing prices
scaled_data = scale(boston_data_set.data)   # scaling
p_set = linspace(1, 10, num=200)    # slicing interval from 1 to 10

"""Finding the best p for minkowski metrics"""
best_p = -1
max_score = -10000000
k_fold = KFold(n_splits=5, random_state=42, shuffle=True)   
for p_val in p_set:
    kn_regressor = KNeighborsRegressor(n_neighbors=5, weights='distance', p=p_val)  # trying KNeighborsRegressor with p_val
    score = cross_val_score(kn_regressor, X=scaled_data, y=boston_data_set.target, cv=k_fold, scoring='neg_mean_squared_error', verbose=0)  # cross validating algo using KFold
    mean = score.mean()
    # bigger is better
    if mean > max_score:
        max_score = mean
        best_p = p_val

print 'Optimal quality (scaled data):', max_score
print 'Best p for minkowski metrics:', best_p