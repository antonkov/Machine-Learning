import pandas as pd
import matplotlib as mpt
mpt.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

filenames = ['newBasis1', 'newBasis2', 'newBasis3']
fig_idx = 1
ans = [5, 10, 10]

for filename, cur in zip(filenames, [200, 200, 200]):
    data = pd.read_csv(filename, index_col=False, header=None, delimiter=' ')
    data = data.dropna(axis=1) # malformed file with ' ' at end

    pca = PCA(n_components=cur)
    pca.fit(data)
    new_data = pca.transform(data)

    ls = []
    for i in range(1, cur + 1):
        x = 0
        for j in range(i, cur + 1):
            x += 1.0 / cur / j
        ls.append(x)
    for i in range(cur):
        if pca.explained_variance_ratio_[i] < ls[i]:
            print('Broke stick rule says to leave ', i, ' components')
            break

    print(sum(pca.explained_variance_ratio_))
    print('drawing')

    plt.figure(fig_idx, figsize=(4, 3))
    fig_idx += 1
    plt.clf()
    plt.plot(pca.explained_variance_ratio_[0:20], linewidth=2)
    plt.xlabel('ncomponents')
    plt.ylabel('explained variance ratio')
plt.show()
# Accepted