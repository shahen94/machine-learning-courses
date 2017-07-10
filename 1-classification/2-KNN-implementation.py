from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection.tests.test_split import train_test_split
from collections import Counter
import numpy as np

iris = load_iris()

train_feature, test_feature, train_label, test_label = train_test_split(iris.data, iris.target, test_size=.5)

class KNearestNeighboard(object):
    def __init__(self, k):
        self.k = k
    def fit(self, features, labels):
        self.features = features
        self.labels = labels

    def predict(self, test_features):
        predictions = []

        for row in test_features:
            label = self._predictLabel(row)
            predictions.append(label)
        return predictions
    def _predictLabel(self, row):
        distances = []

        for value in self.features:
            distance = np.linalg.norm(row - value)
            distances.append(distance)
        
        k_nearest_distances = sorted(distances)[:self.k]

        labels = []
        for distance in k_nearest_distances:
            labels.append(self.labels[distances.index(distance)])
        
        return Counter(labels).most_common(1)[0][0]


c1 = KNN(n_neighbors=5)
c1.fit(train_feature, train_label)

classifier = KNearestNeighboard(k=5)
classifier.fit(train_feature, train_label)

print "Accuracy of sklearn impl: {}".format(accuracy_score(test_label, c1.predict(test_feature)))
print "Accuracy of custom impl: {}".format(accuracy_score(test_label, classifier.predict(test_feature)))