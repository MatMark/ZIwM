import numpy as np

clfs = (
    '256layers_momentum',
    '512layers_momentum',
    '1024layers_momentum',
    '256layers_without',
    '512layers_without',
    '1024layers_without',
)
features = 31


all_score = np.zeros((features, 6))
for i in range(1, features + 1):
    scores = np.load('results/results_' + str(i) + '.npy')
    mean_score = np.mean(scores, axis=1)
    for clf_id, clf_name in enumerate(clfs):
        all_score[i-1][clf_id] = mean_score[clf_id]

# print(all_score)
np.savetxt("results/means.csv", all_score, delimiter=" & ", fmt='%.3f')
np.savetxt("results/means_full.csv", all_score, delimiter=" & ")

max_index_col = np.argmax(all_score, axis=0)
max_index_col = np.reshape(max_index_col, (1, 6))

max = all_score.max(axis=0)
max = np.reshape(max, (1, 6))

max = np.append(max_index_col, max, axis=0)

np.savetxt("results/best_indexes.csv", max_index_col, delimiter=" & ", fmt='%d')
np.savetxt("results/best.csv", max, delimiter=" & ", fmt='%.3f')
np.savetxt("results/best_full.csv", max, delimiter=" & ")

