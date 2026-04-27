'''
*** Logistic Regression in sklearn***
*** iris dataset
*** Machine Learning Mar/2024
*** Ehab M. Iqnaibi
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression

def roc_plot(score, test):
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(test.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(test[:, i], score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i in range(test.shape[1]):
        plt.plot(fpr[i], tpr[i], label=f'ROC curve (area = {roc_auc[i]:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

# Load data
iris = load_iris()
X, y = iris.data, iris.target
y = label_binarize(y, classes=[0, 1, 2])

# Shuffle the data
random_indices = np.random.permutation(len(X))
X_shuffled = X[random_indices]
y_shuffled = y[random_indices]

# Split data into 3 sets
set1_X, set2_X, set3_X = np.array_split(X_shuffled, 3)
set1_y, set2_y, set3_y = np.array_split(y_shuffled, 3)

# Perform threefold cross-validation
for X_test, y_test, X_train, y_train in [(set1_X, set1_y, np.vstack([set2_X, set3_X]), np.vstack([set2_y, set3_y])),
                                         (set2_X, set2_y, np.vstack([set1_X, set3_X]), np.vstack([set1_y, set3_y])),
                                         (set3_X, set3_y, np.vstack([set1_X, set2_X]), np.vstack([set1_y, set2_y]))]:
    # Initialize and train logistic regression model
    model = LogisticRegression(multi_class='ovr', solver='lbfgs', max_iter=10000)
    model.fit(X_train, np.argmax(y_train, axis=1))  # np.argmax converts back to single column labels

    # Make predictions
    y_score = model.predict_proba(X_test)  # Predict probabilities for ROC curve

    # Plot ROC curve
    roc_plot(y_score, y_test)

