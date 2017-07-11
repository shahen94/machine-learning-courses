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


class MeanShift(object):
    def __init__(self, bandwidth):
        self.bandwidth = bandwidth
        self.centroids = {}

    def fit(self, data):
        self.centroids = {}

        # Every item is a centroid
        for i in range(len(data)):
            self.centroids[i] = data[i]
        
        while True:
            new_centroids = []

            for i in self.centroids:
                in_bandwidth = []
                curr_centroid = self.centroids[i]


                for features in data:
                    # is in radius or not
                    if np.linalg.norm(features - curr_centroid) < self.bandwidth:
                        in_bandwidth.append(features)
                        self.classification[i].append(features)
                
                new_centroid = np.average(in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid))

            uniques = sorted(list(set(new_centroids)))

            prev_centroids = dict(self.centroids)

            self.centroids = {}

            for i in range(len(uniques)):
                self.centroids[i] = np.array(uniques[i])
            
            optimized = True

            for i in self.centroids:
                if not np.array_equal(self.centroids[i], prev_centroids[i]):
                    optimized = False
                if not optimized:
                    break
            
            if optimized:
                break

clst = MeanShift(bandwidth=4)

clst.fit(dataset)

plt.scatter(dataset[:, 0], dataset[:, 1])

for i in clst.centroids:
    plt.scatter(clst.centroids[i][0], clst.centroids[i][1], marker='x')

plt.show()