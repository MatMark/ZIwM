import pandas as pd
import numpy as np
import sklearn.feature_selection as fs


def main():
    x, y = load_data()
    result, support = feature_selection(x, y)
    np.savetxt("output.csv", result, delimiter=",", fmt="%d & %0.7f")


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
    scores = []

    for j in range(len(fit.scores_)):
        scores.append((j+1, fit.scores_[j]))
        scores.sort(key=lambda tup: tup[1], reverse=True)
    return scores, selector.get_support()


main()
