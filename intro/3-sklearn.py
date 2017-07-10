from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection.tests.test_split import train_test_split
import numpy

# Load huge dataset into dataset variable
dataset = load_iris()

print dataset.data # Features
print dataset.target # label for each feature
print dataset.target_names # Label names

# Separate data 2 parts, 1. For train 2. For testing

train_features, test_features, train_labels, test_labels = train_test_split(dataset.data, dataset.target, test_size=.5)

# Create classifier instance
classifier = tree.DecisionTreeClassifier()

# Fit data and train
classifier.fit(train_features, train_labels)

# Now let's predict,
# Features from https://ru.wikipedia.org/wiki/%D0%98%D1%80%D0%B8%D1%81%D1%8B_%D0%A4%D0%B8%D1%88%D0%B5%D1%80%D0%B0
print classifier.predict([4.4, 3.0, 1.3, 0.2])

# Check accuracy
print "Accuracy: {}".format(accuracy_score(test_labels, classifier.predict(test_features)))
