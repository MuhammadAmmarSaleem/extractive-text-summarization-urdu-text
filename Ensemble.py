from nltk.classify.svm import SvmClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, ShuffleSplit, train_test_split
from sklearn import preprocessing
from xgboost import XGBRegressor

from utilities import *
import numpy as np
from sklearn import linear_model, __all__
from sklearn.svm import SVR
# from Learn import normalize_dataset, evaluate_summarizer
from Learn import evaluate_summarizer
import urduhack
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn import svm
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor

import lightgbm as lgb


def Ensemble_Model(knn_best,rf_best,log_reg,sv_reg, samples, targets, labels, dataset, feature_names, ur_cue_words):
    print(samples.shape)
    print(len(targets))
    rouge_scores = {
        'rouge-1': {'p': [], 'f': [], 'r': []},
        'rouge-2': {'p': [], 'f': [], 'r': []},
        'rouge-l': {'p': [], 'f': [], 'r': []}
    }

    # create a dictionary of our models
    estimators = [('knn', knn_best), ('rf', rf_best),
                  ('log_reg', log_reg), ('sv_reg', sv_reg)]  # create our voting classifier, inputting our models
    ensemble = VotingClassifier(estimators, voting='hard')
    ensemble.fit(samples, labels)

    train_data = lgb.Dataset(samples, label=labels)
    params = {'learning_rate': 0.001}
    model = lgb.train(params, train_data, 100)
    # model = GradientBoostingRegressor()
    # model.fit(samples, targets)

    # clf.fit(samples, labels)
    rouge_score = evaluate_summarizer(model, dataset, feature_names, ur_cue_words, True)
    for test_type in rouge_scores:
        for param in rouge_scores[test_type]:
            rouge_scores[test_type][param].append(rouge_score[test_type][param])

    avg_scores = {
        'rouge-1': {'p': 0, 'f': 0, 'r': 0},
        'rouge-2': {'p': 0, 'f': 0, 'r': 0},
        'rouge-l': {'p': 0, 'f': 0, 'r': 0}
    }
    for test_type in rouge_scores:
        for param in rouge_scores[test_type]:
            avg_scores[test_type][param] = np.array(rouge_scores[test_type][param]).mean()
    result = {
        # 'mse': mse.mean(),
        # 'r2': scores['test_r2'].mean(),
        'rouge': avg_scores
    }
    return result


def SVR_model(clf, samples, targets, labels, dataset, feature_names, ur_cue_words):
    print(samples.shape)
    print(len(targets))
    rouge_scores = {
        'rouge-1': {'p': [], 'f': [], 'r': []},
        'rouge-2': {'p': [], 'f': [], 'r': []},
        'rouge-l': {'p': [], 'f': [], 'r': []}
    }

    clf.fit(samples, labels)
    rouge_score = evaluate_summarizer(clf, dataset, feature_names, ur_cue_words, True)
    for test_type in rouge_scores:
        for param in rouge_scores[test_type]:
            rouge_scores[test_type][param].append(rouge_score[test_type][param])

    avg_scores = {
        'rouge-1': {'p': 0, 'f': 0, 'r': 0},
        'rouge-2': {'p': 0, 'f': 0, 'r': 0},
        'rouge-l': {'p': 0, 'f': 0, 'r': 0}
    }
    for test_type in rouge_scores:
        for param in rouge_scores[test_type]:
            avg_scores[test_type][param] = np.array(rouge_scores[test_type][param]).mean()
    result = {
        # 'mse': mse.mean(),
        # 'r2': scores['test_r2'].mean(),
        'rouge': avg_scores
    }
    return result


def k_fold_evaluate(clf, samples, targets, dataset, feature_names, ur_cue_words):
    num_folds = 4
    cv = ShuffleSplit(n_splits=num_folds, test_size=0.25, random_state=1)
    scoring = ['neg_mean_squared_error', 'r2']
    scores = cross_validate(clf, samples, targets, cv=cv, scoring=scoring, return_train_score=True,
                            return_estimator=True)
    mse = -1 * scores['test_neg_mean_squared_error']
    rouge_scores = {
        'rouge-1': {'p': [], 'f': [], 'r': []},
        'rouge-2': {'p': [], 'f': [], 'r': []},
        'rouge-l': {'p': [], 'f': [], 'r': []}
    }
    i = 0
    for fitted_clf in scores['estimator']:
        i += 1
        # print("Summarizing dataset by model %d and evaluating ROUGE " % i)
        rouge_score = evaluate_summarizer(fitted_clf, dataset, feature_names, ur_cue_words, True)
        # print_rouges(rouge_score)
        for test_type in rouge_scores:
            for param in rouge_scores[test_type]:
                rouge_scores[test_type][param].append(rouge_score[test_type][param])

    avg_scores = {
        'rouge-1': {'p': 0, 'f': 0, 'r': 0},
        'rouge-2': {'p': 0, 'f': 0, 'r': 0},
        'rouge-l': {'p': 0, 'f': 0, 'r': 0}
    }
    for test_type in rouge_scores:
        for param in rouge_scores[test_type]:
            avg_scores[test_type][param] = np.array(rouge_scores[test_type][param]).mean()
    result = {
        'mse': mse.mean(),
        'r2': scores['test_r2'].mean(),
        'rouge': avg_scores
    }
    return result
    # print("MSE: %0.5f (+/- %0.5f)" % (mse.mean(), mse.std() * 2))
    # print(scores)


def paper_evaluate():

    all_features, targets, labels = load_dataset('features.json')

    all_features = np.array(all_features)
    normalize_dataset(all_features, all_feature_names, 'learn')

    selected_features = select_features(selected_feature_names, all_features)

    # clf = SVR(verbose=False, epsilon=0.01, gamma='auto')
    # clf = LogisticRegression(solver='newton-cholesky')
    documents = urduhack.utils.pickle_load("corpures_preprocessed_test")
    ur_cue_words = read_file("resources/cuewords_ur").split("\n")[:-1]

    # create new a knn model
    knn = KNeighborsClassifier()  # create a dictionary of all values we want to test for n_neighbors
    params_knn = {'n_neighbors': np.arange(1, 25)}  # use gridsearch to test all values for n_neighbors
    knn_gs = GridSearchCV(knn, params_knn, cv=5)  # fit model to training data
    knn_gs.fit(selected_features, labels)

    # save best model
    knn_best = knn_gs.best_estimator_

    # create a new random forest classifier
    rf = RandomForestClassifier()  # create a dictionary of all values we want to test for n_estimators
    params_rf = {'n_estimators': [50, 100, 200]}  # use gridsearch to test all values for n_estimators
    rf_gs = GridSearchCV(rf, params_rf, cv=5)  # fit model to training data
    rf_gs.fit(selected_features, labels)

    # save best model
    rf_best = rf_gs.best_estimator_

    # create a new logistic regression model
    log_reg = LogisticRegression()  # fit the model to the training data
    log_reg.fit(selected_features, labels)

    sv_reg = svm.SVC()
    sv_reg.fit(selected_features, labels)


    exp2_result = Ensemble_Model(knn_best,rf_best,log_reg, sv_reg, selected_features, targets, labels, documents, selected_feature_names, ur_cue_words)
    # exp2_result = SVR_model(clf, selected_features, targets, labels, documents, selected_feature_names, ur_cue_words)

    # exp2_result = k_fold_evaluate(clf, selected_features, labels, documents, selected_feature_names, ur_cue_words)

    # print("MSE: %.5f" % exp2_result['mse'])
    # print('R2 score: %.5f' % exp2_result['r2'])

    # print("MSE on train: %.5f"  % mean_squared_error(y_balanced, y_pred_train))
    # print('Variance score on train: %.5f' % r2_score(y_balanced, y_pred_train))

    print_rouges(exp2_result['rouge'])


paper_evaluate()
