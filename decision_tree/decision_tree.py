import pandas
import pydotplus
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


def main():
    data = pandas.read_csv('titanic.csv', index_col='PassengerId')
    selected_cols = data.loc[:, ['Pclass', 'Fare', 'Age', 'Sex', 'Survived']]

    data_clean_up(selected_cols)

    survived_class = selected_cols['Survived']
    selected_cols.drop('Survived', inplace=True, axis=1)

    clf = DecisionTreeClassifier(random_state=241)
    clf.fit(selected_cols, survived_class)

    print clf.feature_importances_

    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("survived_on_titanic.pdf")


def data_clean_up(data_frame):
    sex_to_float(data_frame)
    data_frame.dropna(axis=0, inplace=True)


def sex_to_float(data_frame):
    data_frame.loc[data_frame.Sex != 'male', 'Sex'] = 0
    data_frame.loc[data_frame.Sex == 'male', 'Sex'] = 1
    return data_frame

main()

