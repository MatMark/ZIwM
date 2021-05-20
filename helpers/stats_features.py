from os import fstat
import numpy as np
from scipy.stats import ttest_ind
from tabulate import tabulate

clfs = (
    '256layers_momentum',
    '512layers_momentum',
    '1024layers_momentum',
    '256layers_without',
    '512layers_without',
    '1024layers_without',
)
num_of_fetaures = 31
f_stat = np.empty((num_of_fetaures, 6), dtype='object')
for index in range(1, num_of_fetaures + 1):
    scores = np.load('results/results_' + str(index) + '.npy')
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
    # t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
    # p_value_table = np.concatenate((names_column, p_value), axis=1)
    # p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    # print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)
    # print("t-statistic:\n", t_statistic_table)
    # with open('results/t_statistic_table.txt', 'w') as f:
    #     f.write(t_statistic_table)

    advantage = np.zeros((len(clfs), len(clfs)))
    advantage[t_statistic > 0] = 1
    # advantage_table = tabulate(np.concatenate(
    #     (names_column, advantage), axis=1), headers)
    # print("Advantage:\n", advantage_table)

    significance = np.zeros((len(clfs), len(clfs)))
    significance[p_value <= alfa] = 1
    # significance_table = tabulate(np.concatenate(
    #     (names_column, significance), axis=1), headers)
    # print("Statistical significance (alpha = 0.05):\n", significance_table)

    stat_better = significance * advantage
    # stat_better_table = tabulate(np.concatenate(
    #     (names_column, stat_better), axis=1), headers)
    # print("Statistically significantly better:\n", stat_better_table)
    # with open('results/stat_better_table.txt', 'w') as f:
    #     f.write(stat_better_table)

    # print(index)
    for clf_id, clf in enumerate(stat_better):
        have = False
        list = []
        for col_id, bet in enumerate(clf):
            if (int(bet) == 1):
                list.append(col_id + 1)
                # print(col_id + 1)
                have = True
        if have:
            # print(clf_id + 1, ": ", list)
            f_stat[index-1][clf_id] = str(list)
        else:
            # print(clf_id + 1, ": ---")
            f_stat[index-1][clf_id] = "---"
print(f_stat)
np.savetxt("results/f_stat.csv", f_stat, delimiter=" & ", fmt='\\tiny %s')