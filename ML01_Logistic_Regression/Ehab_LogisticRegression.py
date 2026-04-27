'''
*** Logistic Regression ***
*** iris dataset
*** Machine Learning Mar/2024
*** Ehab M. Iqnaibi
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
def logistic_regression(X_train, y_train, alpha=0.01, tol=1e-5):
    # m: No.of samples, n: No. of features
    m, n = X_train.shape
    n_classes = y_train.shape[1]
    prev_loss = float('inf')
    theta = np.zeros((n + 1, n_classes))  # Adding one for bias
    X_train_with_bias = np.hstack((np.ones((m, 1)), X_train))  # Add bias term to features
    while True:
        # z = θ^T X , x[100x4] θ[4x3]
        z = np.dot(X_train_with_bias, theta)
        # sigmoid: g(z) =1/(1+ e^-z)
        h = sigmoid(z)
        # Gradient Descent
        # J'(θ) = [( hθ(x(i)) - y(i) ) x_j(i) ]/m
        gradient = np.dot(X_train_with_bias.T, (h - y_train)) / m
        # θ = θj - α J'(θ)
        theta -= alpha * gradient
        current_loss = loss(h, y_train)
        # Check convergence
        if abs(current_loss - prev_loss) < tol:
            break
        prev_loss = current_loss
    return theta

def predict(X_test, theta):
    m = X_test.shape[0]
    X_test_with_bias = np.hstack((np.ones((m, 1)), X_test))  # Add bias term to test features
    z = np.dot(X_test_with_bias, theta)
    return sigmoid(z)

def roc_plot(score, test):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = test.shape[1]
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(test[:, i], score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

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
    # Train logistic regression model
    theta = logistic_regression(X_train, y_train)

    # Make predictions
    y_score = predict(X_test, theta)

    # Plot ROC curve
    roc_plot(y_score, y_test)

