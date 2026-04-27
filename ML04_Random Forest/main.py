'''
Machine Learning , Mar, 2024
Ehab Iqnaibi

Random Forest
n_estimatorsint, default=100 ,The number of trees in the forest.

max_depthint, default=None
The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain
less than min_samples_split samples.

oob_scorebool or callable, default=False
Whether to use out-of-bag samples to estimate the generalization score. By default, accuracy_score is used.
Provide a callable with signature metric(y_true, y_pred) to use a custom metric. Only available if bootstrap=True.

min_samples_splitint or float, default=2
The minimum number of samples required to split an internal node:
'''
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Shuffle
idx = np.arange(X.shape[0])
np.random.shuffle(idx)
X = X[idx]
y = y[idx]

# Create a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=60, max_depth=6, oob_score=True, random_state=10)

# Train the classifier
clf.fit(X, y)

y_pred = clf.predict(X)

# Calculate accuracy
accuracy = accuracy_score(y, y_pred)
print("Accuracy:", accuracy)

# Test dataset score
oob_score = clf.oob_score_
print("Test Score", oob_score)
