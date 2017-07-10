from sklearn.cluster import MeanShift
import numpy as np
import matplotlib.pyplot as plt

dataset = np.array([
    [1, 2],
    [1.5, 1.9],
    [2, 5],
    [14, 3],
    [11, 2],
    [7, 9],
    [7, 11],
    [11, 4]
])

clf = MeanShift(bandwidth=4)
clf.fit(dataset)
for centroid in clf.cluster_centers_:
    plt.scatter(centroid[0], centroid[1], marker='x')

plt.scatter(dataset[:, 0], dataset[:, 1])

plt.show()