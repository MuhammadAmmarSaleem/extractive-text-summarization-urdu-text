from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, ShuffleSplit, train_test_split
from sklearn import preprocessing
from utilities import *
import numpy as np
from sklearn import linear_model, __all__
from sklearn.svm import SVR
# from Learn import normalize_dataset, evaluate_summarizer
from Learn import evaluate_summarizer
import urduhack


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
    clf = LogisticRegression(solver='newton-cholesky')
    documents = urduhack.utils.pickle_load("corpures_preprocessed_test")
    ur_cue_words = read_file("resources/cuewords_ur").split("\n")[:-1]

    exp2_result = SVR_model(clf, selected_features, targets, labels, documents, selected_feature_names, ur_cue_words)

    # exp2_result = k_fold_evaluate(clf, selected_features, labels, documents, selected_feature_names, ur_cue_words)

    # print("MSE: %.5f" % exp2_result['mse'])
    # print('R2 score: %.5f' % exp2_result['r2'])

    # print("MSE on train: %.5f"  % mean_squared_error(y_balanced, y_pred_train))
    # print('Variance score on train: %.5f' % r2_score(y_balanced, y_pred_train))

    print_rouges(exp2_result['rouge'])


paper_evaluate()
