'''import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

# Sample data
x1_data = np.array([-8.0, -6.00, -7.0, -4.0, 2.0, 5.0, 1.0, 3.0, 7.0])
x2_data = np.array([-8.0, -3.0, 2.0, 4.0, 3.0, 7.0, -1.0, -4.0, -7.0])
y_data = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0])

# Reshape data for training
X_train = np.column_stack((x1_data, x2_data))

# Train the Gaussian Naive Bayes classifier
model = GaussianNB()
model.fit(X_train, y_data)

# Plot the decision boundary
h = .02  # step size in the mesh
x_min, x_max = x1_data.min() - 1, x1_data.max() + 1
y_min, y_max = x2_data.min() - 1, x2_data.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(x1_data, x2_data, c=y_data, s=50, edgecolors='k')
plt.title('Gaussian Naive Bayes Classifier')
plt.xlabel('x1_data')
plt.ylabel('x2_data')

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

# Sample data
x1_data = np.array([-8.0, -6.00, -7.0, -4.0, 2.0, 5.0, 1.0, 3.0, 7.0])
x2_data = np.array([-8.0, -3.0, 2.0, 4.0, 3.0, 7.0, -1.0, -4.0, -7.0])
y_data = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0])

# Reshape data for training
X_train = np.column_stack((x1_data, x2_data))

# Train the Gaussian Naive Bayes classifier
model = GaussianNB()
model.fit(X_train, y_data)

# Plot the decision boundary
h = .02  # step size in the mesh
x_min, x_max = x1_data.min() - 1, x1_data.max() + 1
y_min, y_max = x2_data.min() - 1, x2_data.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)

# Plotting with shapes
plt.figure()
plt.title('Gaussian Naive Bayes Classifier')
plt.xlabel('x1_data')
plt.ylabel('x2_data')
for i, point in enumerate(X_train):
    marker = 'o' if y_data[i] == -1 else 'x'
    plt.scatter(point[0], point[1], marker=marker, color='blue' if y_data[i] == -1 else 'red')

plt.contourf(xx, yy, Z, alpha=0.4)
plt.show()
# ********************************
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

# Sample data
x1_data = np.array([-8.0, -6.00, -7.0, -4.0, 2.0, 5.0, 1.0, 3.0, 7.0])
x2_data = np.array([-8.0, -3.0, 2.0, 4.0, 3.0, 7.0, -1.0, -4.0, -7.0])
y_data = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0])

# Reshape data for training
X_train = np.column_stack((x1_data, x2_data))

# Train the Gaussian Naive Bayes classifier
model = GaussianNB()
model.fit(X_train, y_data)

# Plot the decision boundary
h = .02  # step size in the mesh
x_min, x_max = x1_data.min() - 1, x1_data.max() + 1
y_min, y_max = x2_data.min() - 1, x2_data.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)

# Plotting with shapes
plt.figure()
plt.title('Gaussian Naive Bayes Classifier')
plt.xlabel('x1_data')
plt.ylabel('x2_data')
for i, point in enumerate(X_train):
    marker = 'o' if y_data[i] == -1 else 'x'
    plt.scatter(point[0], point[1], marker=marker, color='blue' if y_data[i] == -1 else 'red')

plt.contour(xx, yy, Z, levels=[0], colors='green')  # Plot decision boundary
plt.show()'''


import numpy as np
import matplotlib.pyplot as plt

def gaussian_decision_function(x, mean, covariance):
    inv_covariance = np.linalg.inv(covariance)
    diff = x - mean
    # dj(x) = ln P(wj) - 0.5 ln|Cj| - 0.5 [( x-mj)^T Cj^-1 (x-mj)]
    return np.log(1) - 0.5 * np.log(np.linalg.det(covariance)) - 0.5 * np.dot(diff.T, np.dot(inv_covariance, diff))

# Sample data
x1_data = np.array([-8.0, -6.00, -7.0, -4.0, 2.0, 5.0, 1.0, 3.0, 7.0])
x2_data = np.array([-8.0, -3.0, 2.0, 4.0, 3.0, 7.0, -1.0, -4.0, -7.0])
y_data = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0])

# Separate data into two classes
class1_x1 = x1_data[y_data == -1]
class2_x1 = x1_data[y_data == 1]
class1_x2 = x2_data[y_data == -1]
class2_x2 = x2_data[y_data == 1]

# Calculate mean and covariance for each class
mean_class1 = np.array([np.mean(class1_x1), np.mean(class1_x2)])
covariance_class1 = np.cov(class1_x1, class1_x2)
mean_class2 = np.array([np.mean(class2_x1), np.mean(class2_x2)])
covariance_class2 = np.cov(class2_x1, class2_x2)

# Plot the samples
plt.scatter(class1_x1, class1_x2, c='blue', label='Class -1 (o)')
plt.scatter(class2_x1, class2_x2, c='red', label='Class 1 (x)')

# Generate decision boundary
x_min, x_max = min(x1_data-1), max(x1_data+1)
y_min, y_max = min(x2_data-1), max(x2_data+1)
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = np.zeros_like(xx)

for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        point = np.array([xx[i, j], yy[i, j]])
        decision_class1 = gaussian_decision_function(point, mean_class1, covariance_class1)
        decision_class2 = gaussian_decision_function(point, mean_class2, covariance_class2)
        Z[i, j] = decision_class1 - decision_class2

# Plot decision boundary
plt.contour(xx, yy, Z, levels=[0], colors='green')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Hyperplane based on Gaussian classifier')
plt.legend()
plt.show()
