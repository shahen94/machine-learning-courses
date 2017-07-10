from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection.tests.test_split import train_test_split

dataset = load_iris()

train_feature, test_feature, train_label, test_label = train_test_split(dataset.data, dataset.target, test_size=.5)

clf = KNeighborsClassifier(n_neighbors=2)
clf.fit(train_feature, train_label)

# Features from https://ru.wikipedia.org/wiki/%D0%98%D1%80%D0%B8%D1%81%D1%8B_%D0%A4%D0%B8%D1%88%D0%B5%D1%80%D0%B0
print clf.predict([
    [4.4, 3.0, 1.3, 0.2]
])

# Now let's check accuracy
print "Accuracy: {}".format(accuracy_score(test_label, clf.predict(test_feature)))

