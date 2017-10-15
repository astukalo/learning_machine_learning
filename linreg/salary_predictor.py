import pandas
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge


def main():
    data_train = pandas.read_csv('salary-train.csv')
    _prepare_inplace(data_train)

    loc_contract_transformer, desc_transformer = _fit_extractors(data_train)

    data_train_with_mapped_features = _transform_with_extractors(data_train, loc_contract_transformer, desc_transformer)

    # fitting linear regression with L2 regularization
    clf = Ridge(alpha=1.0, random_state=241)
    clf.fit(data_train_with_mapped_features, data_train['SalaryNormalized'])

    data_test = pandas.read_csv('salary-test-mini.csv')
    _prepare_inplace(data_test)

    data_test_with_mapped_features = _transform_with_extractors(data_test, loc_contract_transformer, desc_transformer)

    # predict using linear regression
    clf_predict = clf.predict(data_test_with_mapped_features)
    print 'Predicted salaries:', clf_predict


def _transform_with_extractors(data, loc_contract_transformer, desc_transformer):
    """
    Transform data with feature extractors to convert nominal features into binary
    """
    desc_matrix = desc_transformer.transform(data['FullDescription'])
    loc_contract_matrix = loc_contract_transformer.transform(data[['LocationNormalized', 'ContractTime']].to_dict('records'))

    # join all extracted features (matrices) into one
    return hstack([desc_matrix, loc_contract_matrix])


def _fit_extractors(data):
    # fitting TF-IDF to extract features from text (with lowercasing)
    desc_transformer = TfidfVectorizer(lowercase=True, min_df=5)
    desc_transformer.fit(data['FullDescription'])

    # fitting extractor to transform string features to binary
    loc_contract_transformer = DictVectorizer()
    loc_contract_transformer.fit(data[['LocationNormalized', 'ContractTime']].to_dict('records'))
    return loc_contract_transformer, desc_transformer


def _prepare_inplace(data):
    data.fillna('nan', inplace=True)
    # commented out, because lower casing in transformer
    # data['FullDescription'] = data['FullDescription'].str.lower()
    data['FullDescription'] = data['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)


if __name__ == "__main__":
    # calling the main function
    main()
