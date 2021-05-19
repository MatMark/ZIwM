import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sklearn.feature_selection as fs
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from scipy.stats import ttest_ind
from tabulate import tabulate

clfs = {
    '256layers_momentum': MLPClassifier(hidden_layer_sizes=(256,),
                                        max_iter=5000, nesterovs_momentum=True,
                                        solver='sgd', random_state=1,
                                        momentum=0.9),
    '512layers_momentum': MLPClassifier(hidden_layer_sizes=(512,),
                                        max_iter=5000, nesterovs_momentum=True,
                                        solver='sgd', random_state=1,
                                        momentum=0.9),
    '1024layers_momentum': MLPClassifier(hidden_layer_sizes=(1024,),
                                         max_iter=5000, nesterovs_momentum=True,
                                         solver='sgd', random_state=1,
                                         momentum=0.9),
    '256layers_without': MLPClassifier(hidden_layer_sizes=(256,),
                                       max_iter=5000, solver='sgd', momentum=0,
                                       random_state=1),
    '512layers_without': MLPClassifier(hidden_layer_sizes=(512,),
                                       max_iter=5000, solver='sgd', momentum=0,
                                       random_state=1),
    '1024layers_without': MLPClassifier(hidden_layer_sizes=(1024,),
                                        max_iter=5000, solver='sgd', momentum=0,
                                        random_state=1),
}


def main():
    x, y = load_data()
    _, scores = feature_selection(x, y)
    if len(sys.argv) > 1:
        max_features = int(sys.argv[1])
    else:
        max_features = 31
    if (max_features > 31 or max_features < 1):
        raise ValueError("Must check for at least one feature and max 31")
    train_evaluate(x, y, max_features)
    ttest()


def load_data():
    file = 'data.csv'
    df = pd.read_csv(file, header=None)
    df = df.to_numpy()
    x = df[:, 0:31]  # features columns;
    y = df[:, 31]  # class column;

    return x, y.astype(int)


def feature_selection(x, y, k=31):
    selector = fs.SelectKBest(score_func=fs.chi2, k=k)
    fit = selector.fit(x, y)
    fit_x = selector.transform(x)
    scores = []

    for j in range(len(fit.scores_)):
        scores.append([j+1, fit.scores_[j]])
    scores.sort(key=lambda x: x[1], reverse=True)
    return fit_x, scores


def train_evaluate(x, y, max_features=31):
    mean_scores = np.empty((max_features, (len(clfs))))
    for i in range(1, max_features + 1):
        print(str(i) + " features")
        fit_x, _ = feature_selection(x, y, i)
        kfold = RepeatedStratifiedKFold(
            n_splits=2, n_repeats=5, random_state=1)
        scores = np.zeros((len(clfs), 2*5))

        for fold_id, (train, test) in enumerate(kfold.split(fit_x, y)):
            for clf_id, clf_name in enumerate(clfs):
                clf = clone(clfs[clf_name])
                clf.fit(fit_x[train], y[train])
                prediction = clf.predict(fit_x[test])
                scores[clf_id, fold_id] = accuracy_score(y[test], prediction)
        mean_score = np.mean(scores, axis=1)
        np.save('results/results_' + str(i), scores)
        # only for ploting
        for idx, score in np.ndenumerate(mean_score):
            mean_scores[i-1][idx[0]] = score
        print(str(int((i/max_features)*100)) + "%" + " completed")
    for clf_id, clf_name in enumerate(clfs):
        x_axis_values = []
        for j in range(0, max_features):
            x_axis_values.append(mean_scores[j][clf_id])
        features = list(range(1, max_features + 1))
        plt.plot(features, x_axis_values, label=clf_name,
                 linewidth=1, marker='o', markersize=5)
    plt.xlabel('Feature Count')
    plt.ylabel('Mean Score')
    plt.xlim([0, max_features + 1])
    plt.ylim([0, 1])
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.grid(True)
    plt.grid(which='both')
    plt.grid(which='minor', alpha=0.2)
    plt.grid(which='major', alpha=0.5)
    plt.legend()
    plt.savefig("W_" + str(i) + ".png", dpi=600)
    plt.clf()
    return mean_scores


def ttest():
    scores = np.load('results/results_23.npy')  # have best results
    t_statistic = np.zeros((len(clfs), len(clfs)))
    p_value = np.zeros((len(clfs), len(clfs)))
    alfa = .05

    for i in range(len(clfs)):
        for j in range(len(clfs)):
            t_statistic[i, j], p_value[i, j] = ttest_ind(
                scores[i], scores[j])

    headers = []
    names_column = np.empty(((len(clfs), 1)), dtype='object')
    for clf_id, clf_name in enumerate(clfs):
        headers.append(clf_name)
        names_column[clf_id][0] = clf_name
    t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
    t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

    advantage = np.zeros((len(clfs), len(clfs)))
    advantage[t_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate(
        (names_column, advantage), axis=1), headers)
    print("Advantage:\n", advantage_table)

    significance = np.zeros((len(clfs), len(clfs)))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(np.concatenate(
        (names_column, significance), axis=1), headers)
    print("Statistical significance (alpha = 0.05):\n", significance_table)

    stat_better = significance * advantage
    stat_better_table = tabulate(np.concatenate(
        (names_column, stat_better), axis=1), headers)
    print("Statistically significantly better:\n", stat_better_table)


if __name__ == "__main__":
    main()
