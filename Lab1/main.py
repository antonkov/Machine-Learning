import pandas as pd
import numpy as np
import string
from random_forest import RandomForest
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import f1_score
from sklearn.metrics import mutual_info_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

X_train = pd.read_csv('arcene_train.data', header=None, delimiter=' ').dropna(axis=1)
Y_train = pd.read_csv('arcene_train.labels', header=None)
X_valid = pd.read_csv('arcene_valid.data', header=None, delimiter=' ').dropna(axis=1)
Y_valid = pd.read_csv('arcene_valid.labels', header=None)


def to_array(df):
    return df.squeeze().ravel()


def mutual_info(x, y):
    c_xy = np.histogram2d(x.values.ravel(), y.values.ravel(), 5)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

pearson = np.zeros(X_train.shape[1])
spearman = np.zeros(X_train.shape[1])
mi = np.zeros(X_train.shape[1])
for id in range(len(X_train)):
    X_feature = X_train[id]
    pearson[id] = pearsonr(to_array(X_feature), to_array(Y_train))[0]
    spearman[id] = spearmanr(X_feature, Y_train)[0]
    mi[id] = mutual_info(X_feature, Y_train)

pearson_idx = np.argsort(pearson)[::-1]
spearman_idx = np.argsort(spearman)[::-1]
mi_idx = np.argsort(mi)[::-1]


"""
take = 300
n = 100
img = np.zeros((n, n))


def to_ij(id):
    return id // n, id % n

for i in range(take):
    id_i, id_j = to_ij(pearson_idx[i])
    img[id_i][id_j] += 1
    id_i, id_j = to_ij(spearman_idx[i])
    img[id_i][id_j] += 2

fig = plt.figure(figsize=(6, 3.2))
plt.imshow(img)
plt.show()
print(pearson[pearson_idx])
print(spearman[spearman_idx])
"""


X_train = X_train.values
X_valid = X_valid.values
Y_train = Y_train.values.ravel()
Y_valid = Y_valid.values.ravel()
n_estimators = 101
criterion = 'gini'


def sklearn_random_forest():
    forest = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion)
    forest.fit(X_train, Y_train)
    return forest


def manual_random_forest():
    print('Starting building Random Forest')
    forest = RandomForest(n_estimators=n_estimators, criterion=criterion)
    forest.fit(X_train, Y_train)
    return forest

importances_filename = 'rf_importances'
"""
forest = manual_random_forest()
Y_pred = forest.predict(X_valid)
print('F measure', f1_score(Y_valid, Y_pred))

importances = forest.feature_importances_
np.save(importances_filename, importances)
"""

rf_importances = np.load(importances_filename + '.npy')
rf_idx = np.argsort(rf_importances)[::-1]

# Importances from mutual information
importances = mi
indices = mi_idx


def draw(ids, names, n):
    for i in range(len(ids)):
        ids[i] = ids[i][:n]
    all_selected_features = list(set(np.concatenate(ids)))
    count_selected_features = len(all_selected_features)
    count_selectors = len(names)
    size_x = 2000
    size_y = 2000
    px = np.int(size_x / count_selected_features)
    py = np.int(size_y / count_selectors)
    img = np.zeros((size_x, size_y))
    def draw_pixel(i, j, color):
        sti = i * py
        stj = j * px
        for ii in range(py):
            for jj in range(px):
                img[sti + ii][stj + jj] = color
    for i, selection_id in enumerate(ids):
        for id in selection_id:
            j = -1
            for ii, x in enumerate(all_selected_features):
                if x == id:
                    j = ii
                    break
            if j != -1:
                draw_pixel(i, j, 3)
    plt.figure(figsize=(12, 12))
    plt.xlabel(' ||| '.join(names))
    plt.imshow(img)
    plt.show()

n = 1000

idxes = [pearson_idx, spearman_idx, mi_idx, rf_idx]
selection_names = ['pearson', 'spearman', 'mutual information', 'random forest']
draw(idxes, selection_names, n)
ss = sum(importances)
for i in range(len(importances)):
    importances[i] /= ss

indices = indices[:n]
res = sum(importances[indices])
print(res * 100, '%')

print('Feature ranking:')

for f in range(min(n, 20)):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


names = [
    'Nearest Neighbours',
    'SVM',
    'Decision Tree',
    'Random Forest',
    'AdaBoost'
]


def create_classifiers():
    return [
        KNeighborsClassifier(3),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
    ]

X_train_part = X_train[:, indices]
X_valid_part = X_valid[:, indices]

for name, clf_all, clf_part in zip(names, create_classifiers(), create_classifiers()):
    clf_all.fit(X_train, Y_train)
    clf_part.fit(X_train_part, Y_train)
    score_on_all = clf_all.score(X_valid, Y_valid)
    score_on_part = clf_part.score(X_valid_part, Y_valid)
    print(name)
    print('-- Score on all features: ', score_on_all)
    print('-- Score on best', n, 'features: ', score_on_part)
    print()

plt.figure()
plt.title('Feature importances')
plt.bar(range(n), importances[indices],
       color="r", align="center")
plt.xlim([-1, n])
plt.show()
