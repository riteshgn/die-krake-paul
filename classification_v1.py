import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import time

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

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

print('\n------------- {} -----------------------\n'.format('Naive Bayes'))

t0 = time.time()

# Specify Model
nb_match_model = GaussianNB()

# Cross Validate
cv_results = cross_validate(nb_match_model, X, y, cv=10, return_estimator=True)
max_score = max([(index, score) for index, score in enumerate(cv_results['test_score'])], key=lambda i_s: i_s[1])

t1 = time.time()
print('\nTime taken for model: {} seconds, Cross-validated max score: {}'.format(t1 - t0, max_score))

# Predict and get metrics
nb_match_model = cv_results['estimator'][max_score[0]]
actual_predictions = nb_match_model.predict(predict_X)
print('\nTime taken for prediction: {} seconds, Accuracy: {}'.format(
    time.time() - t1, accuracy_score(predict_y, actual_predictions)))
print(classification_report(predict_y, actual_predictions))
print('\nConfusion Matrix: ')
print(pd.crosstab(predict_y, actual_predictions, rownames=['True'], colnames=['Predicted'], margins=True))

plt.figure()
plot_confusion_matrix(confusion_matrix(predict_y, actual_predictions), classes=['A', 'D', 'H'],
                      title='NB: Confusion matrix, without normalization')

print('\n------------- {} -----------------------\n'.format('SVM'))

t0 = time.time()

# Specify Model
svm_match_model = SVC(kernel='rbf', gamma='scale')

# Cross Validate
cv_results = cross_validate(svm_match_model, X, y, cv=10, return_estimator=True)
max_score = max([(index, score) for index, score in enumerate(cv_results['test_score'])], key=lambda i_s: i_s[1])

t1 = time.time()
print('\nTime taken for model: {} seconds, Cross-validated max score: {}'.format(t1 - t0, max_score))

# Predict and get metrics
svm_match_model = cv_results['estimator'][max_score[0]]
actual_predictions = svm_match_model.predict(predict_X)
print('\nTime taken for prediction: {} seconds, Accuracy: {}'.format(
    time.time() - t1, accuracy_score(predict_y, actual_predictions)))
print(classification_report(predict_y, actual_predictions))
print('\nConfusion Matrix: ')
print(pd.crosstab(predict_y, actual_predictions, rownames=['True'], colnames=['Predicted'], margins=True))

plt.figure()
plot_confusion_matrix(confusion_matrix(predict_y, actual_predictions), classes=['A', 'D', 'H'],
                      title='SVM: Confusion matrix, without normalization')

print('\n------------- {} -----------------------\n'.format('Random Forest'))

t0 = time.time()

# Specify Model
rf_match_model = RandomForestClassifier(n_estimators=500, n_jobs=4, random_state=1)

# Cross Validate
cv_results = cross_validate(rf_match_model, X, y, cv=10, return_estimator=True)
max_score = max([(index, score) for index, score in enumerate(cv_results['test_score'])], key=lambda i_s: i_s[1])

t1 = time.time()
print('\nTime taken for model: {} seconds, Cross-validated max score: {}'.format(t1 - t0, max_score))

# Predict and get metrics
rf_match_model = cv_results['estimator'][max_score[0]]
# print(rf_match_model.feature_importances_)

actual_predictions = rf_match_model.predict(predict_X)
print('\nTime taken for prediction: {} seconds, Accuracy: {}'.format(
    time.time() - t1, accuracy_score(predict_y, actual_predictions)))
print(classification_report(predict_y, actual_predictions))
print('\nConfusion Matrix: ')
print(pd.crosstab(predict_y, actual_predictions, rownames=['True'], colnames=['Predicted'], margins=True))

plt.figure()
plot_confusion_matrix(confusion_matrix(predict_y, actual_predictions), classes=['A', 'D', 'H'],
                      title='RF: Confusion matrix, without normalization')

print('\n------------- {} -----------------------\n'.format('Gradient Boost'))

t0 = time.time()

# Specify Model
gb_match_model = GradientBoostingClassifier(n_estimators=100, random_state=1)

# Cross Validate
cv_results = cross_validate(gb_match_model, X, y, cv=10, return_estimator=True)
max_score = max([(index, score) for index, score in enumerate(cv_results['test_score'])], key=lambda i_s: i_s[1])

t1 = time.time()
print('\nTime taken for model: {} seconds, Cross-validated max score: {}'.format(t1 - t0, max_score))

# Predict and get metrics
gb_match_model = cv_results['estimator'][max_score[0]]
# print(rf_match_model.feature_importances_)

actual_predictions = gb_match_model.predict(predict_X)
print('\nTime taken for prediction: {} seconds, Accuracy: {}'.format(
    time.time() - t1, accuracy_score(predict_y, actual_predictions)))
print(classification_report(predict_y, actual_predictions))
print('\nConfusion Matrix: ')
print(pd.crosstab(predict_y, actual_predictions, rownames=['True'], colnames=['Predicted'], margins=True))

plt.figure()
plot_confusion_matrix(confusion_matrix(predict_y, actual_predictions), classes=['A', 'D', 'H'],
                      title='GDB: Confusion matrix, without normalization')

print('\n------------- {} -----------------------\n'.format('Ada Boost'))

t0 = time.time()

# Specify Model
ab_match_model = AdaBoostClassifier(n_estimators=500, random_state=1)

# Cross Validate
cv_results = cross_validate(ab_match_model, X, y, cv=10, return_estimator=True)
max_score = max([(index, score) for index, score in enumerate(cv_results['test_score'])], key=lambda i_s: i_s[1])

t1 = time.time()
print('\nTime taken for model: {} seconds, Cross-validated max score: {}'.format(t1 - t0, max_score))

# Predict and get metrics
ab_match_model = cv_results['estimator'][max_score[0]]

actual_predictions = ab_match_model.predict(predict_X)
print('\nTime taken for prediction: {} seconds, Accuracy: {}'.format(
    time.time() - t1, accuracy_score(predict_y, actual_predictions)))
print(classification_report(predict_y, actual_predictions))
print('\nConfusion Matrix: ')
print(pd.crosstab(predict_y, actual_predictions, rownames=['True'], colnames=['Predicted'], margins=True))

plt.figure()
plot_confusion_matrix(confusion_matrix(predict_y, actual_predictions), classes=['A', 'D', 'H'],
                      title='ADB: Confusion matrix, without normalization')

# #############################################################################
# Plot feature importance
feature_importance = gb_match_model.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

fig = plt.figure(figsize=(24.8, 24.8))
fig.add_subplot(1, 1, 1)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, list(X.columns), fontsize=12, rotation=20)
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
