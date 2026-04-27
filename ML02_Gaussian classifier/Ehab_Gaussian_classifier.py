'''
*** Gaussian classifier ***
*** Machine Learning Mar/2024
*** Ehab M. Iqnaibi
'''
import numpy as np
import matplotlib.pyplot as plt

def gaussian_decision_function(x, mean, covariance, prior):
    inv_covariance = np.linalg.inv(covariance)
    diff = x - mean
    # dj(x) = ln P(wj) - 0.5 ln|Cj| - 0.5 [( x-mj)^T Cj^-1 (x-mj)]
    return np.log(prior) - 0.5 * np.log(np.linalg.det(covariance)) - 0.5 * np.dot(diff.T, np.dot(inv_covariance, diff))

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
prior_class1 = len(class1_x1) / len(x1_data)

mean_class2 = np.array([np.mean(class2_x1), np.mean(class2_x2)])
covariance_class2 = np.cov(class2_x1, class2_x2)
prior_class2 = len(class2_x1) / len(x1_data)

# Plot the samples
plt.scatter(class1_x1, class1_x2, c='blue', label='Class -1 (o)')
plt.scatter(class2_x1, class2_x2, c='red', label='Class 1 (x)')

# Generate decision boundary
x_min, x_max = min(x1_data - 1), max(x1_data + 1)
y_min, y_max = min(x2_data - 1), max(x2_data + 1)
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = np.zeros_like(xx)

for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        point = np.array([xx[i, j], yy[i, j]])
        decision_class1 = gaussian_decision_function(point, mean_class1, covariance_class1, prior_class1)
        decision_class2 = gaussian_decision_function(point, mean_class2, covariance_class2, prior_class2)
        Z[i, j] = decision_class1 - decision_class2


# Plot decision boundary
plt.contour(xx, yy, Z, levels=[0], colors='green')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Hyperplane based on Gaussian classifier')
plt.legend()
plt.show()
