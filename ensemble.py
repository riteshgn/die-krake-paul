import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import time

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier

# ENSEMBLE_RESULT_LOOKUP = {
#     'YYYY': 'D', 'YYYN': 'H', 'YYNY': 'A', 'YYNN': 'D',
#     'YNYY': 'H', 'YNYN': 'H', 'YNNY': 'D', 'YNNN': 'H',
#     'NYYY': 'A', 'NYYN': 'D', 'NYNY': 'A', 'NYNN': 'A',
#     'NNYY': 'D', 'NNYN': 'H', 'NNNY': 'A', 'NNNN': 'D',
# }

ENSEMBLE_RESULT_LOOKUP = {
    'YYYY': 'D', 'YYYN': 'H', 'YYNY': 'A', 'YYNN': 'D',
    'YNYY': 'H', 'YNYN': 'H', 'YNNY': 'H', 'YNNN': 'H',
    'NYYY': 'A', 'NYYN': 'A', 'NYNY': 'A', 'NYNN': 'A',
    'NNYY': 'D', 'NNYN': 'H', 'NNNY': 'A', 'NNNN': 'D',
}

# ENSEMBLE_RESULT_LOOKUP = {
#     'YYYY': 'H', 'YYYN': 'H', 'YYNY': 'A', 'YYNN': 'H',
#     'YNYY': 'H', 'YNYN': 'H', 'YNNY': 'H', 'YNNN': 'H',
#     'NYYY': 'A', 'NYYN': 'A', 'NYNY': 'A', 'NYNN': 'A',
#     'NNYY': 'D', 'NNYN': 'H', 'NNNY': 'A', 'NNNN': 'D',
# }

source_matches = [
    {'season': '2008/2009', 'league_id': 1729},
    {'season': '2009/2010', 'league_id': 1729},
    {'season': '2010/2011', 'league_id': 1729},
    {'season': '2011/2012', 'league_id': 1729},
    {'season': '2012/2013', 'league_id': 1729},
    {'season': '2013/2014', 'league_id': 1729},
    {'season': '2014/2015', 'league_id': 1729}
]
filename = 'combined_db/features_{season}_{league}_{version}.csv'

match_data = pd.concat(
    [pd.read_csv(
        filename.format(season=matches['season'].replace('/', '_'), league=matches['league_id'], version='v2_all')
    ) for matches in source_matches],
    axis=0, ignore_index=True)

predict_data = pd.read_csv(filename.format(season='2015_2016', league=1729, version='v2_all'))

print('Train dataset shape: ', match_data.shape)

# Create X
# print(list(match_data.columns[1:-1]))
features_to_drop = ['hp{index}_id'.format(index=plyr_indx) for plyr_indx in range(1, 12)]
features_to_drop = features_to_drop + ['hp{index}_bday'.format(index=plyr_indx) for plyr_indx in range(1, 12)]
features_to_drop = features_to_drop + ['ap{index}_id'.format(index=plyr_indx) for plyr_indx in range(1, 12)]
features_to_drop = features_to_drop + ['ap{index}_bday'.format(index=plyr_indx) for plyr_indx in range(1, 12)]
features_to_drop = features_to_drop + ['Unnamed: 0', 'id', 'ps_date_x', 'ps_date_y', 'result',
                                       'home_team', 'away_team', 'match_date']
features_to_drop = features_to_drop + ['Unnamed: 0.1', 'gameweek', 'match_date.1',
                                       'home_team.1', 'away_team.1']
X = match_data.drop(columns=features_to_drop)
predict_X = predict_data.drop(columns=features_to_drop)
# print(list(X.columns))


def _convert_to_home_win_or_not(elem):
    return 'Y' if elem == 'H' else 'N'


def _convert_to_away_win_or_not(elem):
    return 'Y' if elem == 'A' else 'N'


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    ref: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def _predict_using_naive_bayes(convert_result):
    print('\n------------- {} -----------------------\n'.format('Naive Bayes'))
    # Create target object and call it y
    y = match_data.result.apply(convert_result)
    predict_y = predict_data.result.apply(convert_result)

    # Specify Model
    nb_match_model = GaussianNB()

    # Cross Validate
    cv_results = cross_validate(nb_match_model, X, y, cv=10, return_estimator=True)
    max_score = max([(index, score) for index, score in enumerate(cv_results['test_score'])], key=lambda i_s: i_s[1])
    # print('\nCross-validated max score:', max_score)

    # Predict and get metrics
    nb_match_model = cv_results['estimator'][max_score[0]]
    actual_predictions = nb_match_model.predict(predict_X)
    print('\nAccuracy: ', accuracy_score(predict_y, actual_predictions))
    print(classification_report(predict_y, actual_predictions))
    print('\nConfusion Matrix: ')
    print(pd.crosstab(predict_y, actual_predictions, rownames=['True'], colnames=['Predicted'], margins=True))
    return actual_predictions


def _predict_using_random_forest(convert_result):
    print('\n------------- {} -----------------------\n'.format('Ada Boost'))
    # Create target object and call it y
    y = match_data.result.apply(convert_result)
    predict_y = predict_data.result.apply(convert_result)

    # Specify Model
    rf_match_model = AdaBoostClassifier(n_estimators=250, random_state=1)

    # Cross Validate
    cv_results = cross_validate(rf_match_model, X, y, cv=2, return_estimator=True)
    max_score = max([(index, score) for index, score in enumerate(cv_results['test_score'])], key=lambda i_s: i_s[1])
    # print('\nCross-validated max score:', max_score)

    # Predict and get metrics
    rf_match_model = cv_results['estimator'][max_score[0]]
    # print(rf_match_model.feature_importances_)

    actual_predictions = rf_match_model.predict(predict_X)
    print('\nAccuracy: ', accuracy_score(predict_y, actual_predictions))
    print(classification_report(predict_y, actual_predictions))
    print('\nConfusion Matrix: ')
    print(pd.crosstab(predict_y, actual_predictions, rownames=['True'], colnames=['Predicted'], margins=True))
    return actual_predictions


def _ensemble_result(classifier_predictions):
    key = ''.join(classifier_predictions)
    classifier_predictions[4] = ENSEMBLE_RESULT_LOOKUP[key]
    return classifier_predictions


def predict_games():
    t0 = time.time()

    y_home_nb = _predict_using_naive_bayes(_convert_to_home_win_or_not)
    y_away_nb = _predict_using_naive_bayes(_convert_to_away_win_or_not)
    y_home_rf = _predict_using_random_forest(_convert_to_home_win_or_not)
    y_away_rf = _predict_using_random_forest(_convert_to_away_win_or_not)

    ensemble_predictions = pd.concat([pd.Series(y_home_nb), pd.Series(y_away_nb),
                                     pd.Series(y_home_rf), pd.Series(y_away_rf)], axis=1)
    # print(ensemble_predictions.head(n=10))

    ensemble_predictions_dash = ensemble_predictions.apply(_ensemble_result, axis=1)
    # print(ensemble_predictions_dash.head(n=10))

    t1 = time.time()

    predict_y = predict_data.result
    # print(predict_y.head(n=10))
    actual_predictions = ensemble_predictions_dash[4]
    # print(actual_predictions.head(n=10))

    print('\n------------- {} -----------------------\n'.format('Ensemble'))

    print('\nTime taken: {} seconds, Accuracy: {}'.format(t1-t0, accuracy_score(predict_y, actual_predictions)))
    print(classification_report(predict_y, actual_predictions))
    print('\nConfusion Matrix: ')
    print(pd.crosstab(predict_y, actual_predictions, rownames=['True'], colnames=['Predicted'], margins=True))

    plt.figure()
    plot_confusion_matrix(confusion_matrix(predict_y, actual_predictions), classes=['A', 'D', 'H'],
                          title='Confusion matrix, without normalization')
    plt.show()


if __name__ == '__main__':
    predict_games()
