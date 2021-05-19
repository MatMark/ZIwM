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

scores27 = np.load('results/results_27.npy')
scores28 = np.load('results/results_28.npy')
scores29 = np.load('results/results_29.npy')

# get best scores
scores = np.zeros((6, 2*5))
scores[0] = scores27[0]
scores[1] = scores28[1]
scores[2] = scores28[2]
scores[3] = scores29[3]
scores[4] = scores28[4]
scores[5] = scores29[5]

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
# p_value_table = np.concatenate((names_column, p_value), axis=1)
# p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
# print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)
print("t-statistic:\n", t_statistic_table)
with open('results/t_statistic_table.txt', 'w') as f:
    f.write(t_statistic_table)

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
stat_better_table = tabulate(np.concatenate(
    (names_column, stat_better), axis=1), headers)
print("Statistically significantly better:\n", stat_better_table)
with open('results/t_statistic_table.txt', 'w') as f:
    f.write(stat_better_table)
