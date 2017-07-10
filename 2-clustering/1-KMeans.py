import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

dataset = np.array([
    [1, 2],
    [1.5, 2],
    [3, 1],
    [12, 10],
    [9, 12],
    [7, 11]
])
clf = KMeans(n_clusters=2)
clf.fit(dataset)
colors = ['g', 'r']
print clf.labels_
for centroid in clf.cluster_centers_:
    plt.scatter(centroid[0], centroid[1], marker='x')

for idx, data in enumerate(dataset):
    plt.scatter(data[0], data[1], color=colors[clf.labels_[idx]])


plt.show()