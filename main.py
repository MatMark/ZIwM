import xlsxwriter as xls
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sklearn.feature_selection as fs
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier

best_conf_matrix = [0, False, 0, 0, np.ndarray, 0]


def main():
    x, y = load_data()
    _, scores = feature_selection(x, y)
    if len(sys.argv) > 1:
        max_features = int(sys.argv[1])
    else:
        max_features = 31
    if (max_features > 31 or max_features < 1):
        raise ValueError("Must check for at least one feature and max 31")

    with xls.Workbook('ranking.xlsx') as workbook:
        worksheet = workbook.add_worksheet()
        for row_num, data in enumerate(scores):
            worksheet.write_row(row_num, 0, data)

    hidden_layer_width = [256, 512, 1024]
    momentum = [0, 0.35, 0.65, 0.95]
    features = list(range(1, max_features + 1))

    for width in hidden_layer_width:
        print("Hidden layer width: " + str(width))
        scores = []
        workbook = xls.Workbook("scores" + str(width) + ".xlsx")
        worksheet = workbook.add_worksheet()
        worksheet.write(0, 0, "Feature Count")
        worksheet.write_column(1, 0, features)
        col = 1
        for mc in momentum:
            print("Momentum coefficient: " + str(mc))
            if (mc == 0):
                print("Momentum: " + str(False))
                scores = train_evaluate(x, y, width, False, mc, max_features)
                data_label = "M: NO"
            else:
                print("Momentum: " + str(True))
                scores = train_evaluate(x, y, width, True, mc, max_features)
                data_label = "M: " + str(mc)
            worksheet.write(0, col, "M: " + str(mc))
            worksheet.write_column(1, col, scores)
            plt.plot(features, scores, label=data_label,
                     linewidth=1, marker='o', markersize=5)
            col += 1

        plt.title("Hidden layer width: " + str(width))
        plt.xlabel('Feature Count')
        plt.ylabel('Mean Score')
        plt.xlim([0, max_features + 1])
        plt.ylim([0, 100])
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
        plt.grid(True)
        plt.grid(which='both')
        plt.grid(which='minor', alpha=0.2)
        plt.grid(which='major', alpha=0.5)
        plt.legend()
        plt.savefig("W_" + str(width) + ".png", dpi=600)
        plt.clf()
        workbook.close()
    summaries = "Hidden layer width: " + str(best_conf_matrix[0]) + \
                "\nMomentum: " + str(best_conf_matrix[1]) + \
                "\nMomentum coef: " + str(best_conf_matrix[2]) + \
                "\nFeatures number: " + str(best_conf_matrix[3]) + \
                "\nConfusion matrix:\n" + str(best_conf_matrix[4]) + \
                "\nScore: " + str(best_conf_matrix[5])
    print(summaries)
    summary = open("summary.txt", "w+")
    summary.write(summaries)
    summary.close()


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


def train_evaluate(x, y, hidden_layer_width, use_momentum=True, momentum=0.9, max_features=31):
    scores = []
    for i in range(1, max_features + 1):
        global best_conf_matrix
        fit_x, _ = feature_selection(x, y, i)
        kfold = RepeatedStratifiedKFold(
            n_splits=2, n_repeats=5, random_state=None)
        if use_momentum:
            mlp = MLPClassifier(hidden_layer_sizes=(hidden_layer_width,),
                                max_iter=5000, nesterovs_momentum=True,
                                solver='sgd', random_state=1,
                                momentum=momentum)
        else:
            mlp = MLPClassifier(hidden_layer_sizes=(hidden_layer_width,),
                                max_iter=5000, solver='sgd', momentum=0,
                                random_state=1)
        val_acc_features = []

        for train_index, test_index in kfold.split(fit_x, y):
            x_train, x_test = fit_x[train_index], fit_x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            mlp.fit(x_train, y_train)

            prediction = mlp.predict(x_test)
            conf_mat = confusion_matrix(y_test, prediction)
            s = mlp.score(x_test, y_test)
            if best_conf_matrix[5] < s:
                best_conf_matrix = [hidden_layer_width, use_momentum,
                                    momentum, i, conf_mat, s]
            val_acc_features.append(s)

        mean_score = np.mean(val_acc_features) * 100
        print("Mean score for " + str(i) + " features: " +
              str(mean_score) + "\n")
        scores.append(mean_score)

    return scores


if __name__ == "__main__":
    main()
