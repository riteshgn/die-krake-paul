import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools


from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB

VERSIONS = {
    'KAGGLE_FEATURES_V1': {'root': 'kaggle_db', 'version': 'v1'},
    'KAGGLE_FEATURES_V2': {'root': 'kaggle_db', 'version': 'v2_all'},
    'COMBINED_FEATURES_V1': {'root': 'combined_db', 'version': 'v1_all'},
    'COMBINED_FEATURES_V2': {'root': 'combined_db', 'version': 'v2_all'},
}

SELECTED_VERSION = 'KAGGLE_FEATURES_V2'

# print(list(match_data.columns[1:-1]))
features_to_drop = ['Unnamed: 0', 'result']

# only for kaggle v1
if SELECTED_VERSION == 'KAGGLE_FEATURES_V1':
    features_to_drop = features_to_drop + ['date_y']

# for kaggle v2 and later
if SELECTED_VERSION != 'KAGGLE_FEATURES_V1':
    features_to_drop = features_to_drop + ['hp{index}_id'.format(index=plyr_indx) for plyr_indx in range(1, 12)]
    features_to_drop = features_to_drop + ['hp{index}_bday'.format(index=plyr_indx) for plyr_indx in range(1, 12)]
    features_to_drop = features_to_drop + ['ap{index}_id'.format(index=plyr_indx) for plyr_indx in range(1, 12)]
    features_to_drop = features_to_drop + ['ap{index}_bday'.format(index=plyr_indx) for plyr_indx in range(1, 12)]
    features_to_drop = features_to_drop + ['id', 'ps_date_x', 'ps_date_y', 'home_team', 'away_team', 'match_date']

# only for combined
if SELECTED_VERSION in ['COMBINED_FEATURES_V1', 'COMBINED_FEATURES_V2']:
    features_to_drop = features_to_drop + ['Unnamed: 0.1', 'gameweek', 'match_date.1', 'home_team.1', 'away_team.1']

source_matches = [
    {'season': '2008/2009', 'league_id': 1729},
    {'season': '2009/2010', 'league_id': 1729},
    {'season': '2010/2011', 'league_id': 1729},
    {'season': '2011/2012', 'league_id': 1729},
    {'season': '2012/2013', 'league_id': 1729},
    {'season': '2013/2014', 'league_id': 1729},
    {'season': '2014/2015', 'league_id': 1729}
]
filename = '{root}/features_{season}_{league}_{version}.csv'

match_data = pd.concat(
    [pd.read_csv(
        filename.format(
            root=VERSIONS[SELECTED_VERSION]['root'],
            season=matches['season'].replace('/', '_'),
            league=matches['league_id'],
            version=VERSIONS[SELECTED_VERSION]['version'])
    ) for matches in source_matches],
    axis=0, ignore_index=True)

predict_data = pd.read_csv(filename.format(
    root=VERSIONS[SELECTED_VERSION]['root'],
    season='2015_2016',
    league=1729,
    version=VERSIONS[SELECTED_VERSION]['version']))

print('Train dataset shape: ', match_data.shape)


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


# match_data = pd.read_csv(match_file_path)
# Create target object and call it y
y = match_data.result
predict_y = predict_data.result

# Create X
# print(features_to_drop)
X = match_data.drop(columns=features_to_drop)
predict_X = predict_data.drop(columns=features_to_drop)
# print(list(X.columns))

print('\n------------- {} -----------------------\n'.format('Naive Bayes'))

# Specify Model
nb_match_model = GaussianNB()

# Cross Validate
cv_results = cross_validate(nb_match_model, X, y, cv=10, return_estimator=True)
max_score = max([(index, score) for index, score in enumerate(cv_results['test_score'])], key=lambda i_s: i_s[1])
print('\nCross-validated max score:', max_score)

# Predict and get metrics
nb_match_model = cv_results['estimator'][max_score[0]]
actual_predictions = nb_match_model.predict(predict_X)
print('\nAccuracy: ', accuracy_score(predict_y, actual_predictions))
print(classification_report(predict_y, actual_predictions))
print('\nConfusion Matrix: ')
print(pd.crosstab(predict_y, actual_predictions, rownames=['True'], colnames=['Predicted'], margins=True))

plt.figure()
plot_confusion_matrix(confusion_matrix(predict_y, actual_predictions), classes=['A', 'D', 'H'],
                      title='Confusion matrix, without normalization')
plt.show()
