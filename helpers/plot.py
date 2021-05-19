import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

x_axis_256_m = []
x_axis_512_m = []
x_axis_1024_m = []
x_axis_256_w = []
x_axis_512_w = []
x_axis_1024_w = []
for i in range(0, features):
    x_axis_256_m.append(all_score[i][0])
    x_axis_512_m.append(all_score[i][1])
    x_axis_1024_m.append(all_score[i][2])
    x_axis_256_w.append(all_score[i][3])
    x_axis_512_w.append(all_score[i][4])
    x_axis_1024_w.append(all_score[i][5])

# with momentum
features_list = list(range(1, features + 1))
plt.plot(features_list, x_axis_256_m, label='256 layers',
         linewidth=1)
plt.plot(features_list, x_axis_512_m, label='512 layers',
         linewidth=1)
plt.plot(features_list, x_axis_1024_m, label='1024 layers',
         linewidth=1)
plt.title("With momentum")
plt.xlabel('Feature Count')
plt.ylabel('Mean Score')
plt.xlim([0, features + 1])
plt.ylim([0.5, 1])
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
plt.gcf().canvas.draw()
tl = plt.gca().get_xticklabels()
maxsize = max([t.get_window_extent().width for t in tl])
m = 0.5  # inch margin
s = maxsize/plt.gcf().dpi*31+2*m
margin = m/plt.gcf().get_size_inches()[0]

plt.gcf().subplots_adjust(left=margin, right=1.-margin)
plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])
plt.grid(True)
plt.grid(which='both')
plt.grid(which='minor', alpha=0.2)
plt.grid(which='major', alpha=0.5)
plt.legend(loc=4)
plt.savefig("results/momentum.png", dpi=1200)
plt.clf()

# without momentum
plt.plot(features_list, x_axis_256_w, label='256 layers',
         linewidth=1)
plt.plot(features_list, x_axis_512_w, label='512 layers',
         linewidth=1)
plt.plot(features_list, x_axis_1024_w, label='1024 layers',
         linewidth=1)
plt.title("Without momentum")
plt.xlabel('Feature Count')
plt.ylabel('Mean Score')
plt.xlim([0, features + 1])
plt.ylim([0.5, 1])
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
plt.gcf().canvas.draw()
tl = plt.gca().get_xticklabels()
maxsize = max([t.get_window_extent().width for t in tl])
m = 0.6  # inch margin
s = maxsize/plt.gcf().dpi*31+2*m
margin = m/plt.gcf().get_size_inches()[0]

plt.gcf().subplots_adjust(left=margin, right=1.-margin)
plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])
plt.grid(True)
plt.grid(which='both')
plt.grid(which='minor', alpha=0.2)
plt.grid(which='major', alpha=0.5)
plt.legend(loc=4)
plt.savefig("results/without.png", dpi=1200)
plt.clf()
