import numpy as np
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

class KMeans(object):
    def __init__(self, n_cluster=1, tolerance=.001, max_iteration=300):
        self.n_cluster = n_cluster
        self.tolerance = tolerance
        self.max_iteration = max_iteration
        self.classification = {}
        self.centroids = {}
        self._iterated = 0

    def fit(self, data):
        # Random
        for i in range(self.n_cluster):
            self.centroids[i] = data[i]
        
        # Classify data
        for i in range(self.max_iteration):
            self._iterated += 1
            self.classification = {}

            for ii in range(self.n_cluster):
                self.classification[ii] = []
            
            # Iterate over dataset and classify it for current centroids
            for featureset in data:

                distances = []
                for centroid in self.centroids:
                    distances.append(np.linalg.norm(featureset - self.centroids[centroid]))
                
                # Get the distance from closest centroid
                closest_dist = min(distances)

                # Get the index (centroid index)
                classification = distances.index(closest_dist)

                # Add current featureset to an centroid array
                self.classification[classification].append(featureset)

            prev_centroid = dict(self.centroids)
            optimized = True

            for classification in self.classification:
                '''
                So we should get average of X and Y coordinates
                Xavg = (X1 + X2 + X3 + ...Xn) / n
                Yavg = (Y1 + Y2 + Y3 + ...Yn) / n

                ^
                | x1
                |---*
                |x2
                |- *
                |x3
                |--*
                0--------------------->
                |
                '''
                self.centroids[classification] = np.average(self.classification[classification], axis=0)

            for centroid in self.centroids:
                orig_cent = prev_centroid[centroid]
                curr_cent = self.centroids[centroid]

                if np.sum((curr_cent - orig_cent) / orig_cent * 100.0) > self.tolerance:
                    optimized = False

            if optimized:
                break

    

clf = KMeans(n_cluster=2, max_iteration=10000)
clf.fit(dataset)
colors = ['g', 'r']

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1])

for classification in clf.classification:
    plt.scatter(clf.classification[classification][0], clf.classification[classification][1], color=colors[classification])

plt.show()