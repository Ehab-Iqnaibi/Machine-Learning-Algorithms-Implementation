import numpy as np

# Sample data
x1_data = np.array([-8.0, -6.00, -7.0, -4.0, 2.0, 5.0, 1.0, 3.0, 7.0])
x2_data = np.array([-8.0, -3.0, 2.0, 4.0, 3.0, 7.0, -1.0, -4.0, -7.0])
y_data = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0])

# Separate data into class A and class B
class_A = np.array([[x1_data[i], x2_data[i]] for i in range(len(y_data)) if y_data[i] == -1])

# Calculate mean vector for class A
mean_A = np.mean(class_A, axis=0)

# Compute covariance matrix for class A
covariance_A = np.zeros((2, 2))
N_A = len(class_A)
for x in class_A:
    x = np.reshape(x, (2, 1))  # Reshape x into a column vector
    mean_A = np.reshape(mean_A, (2, 1))  # Reshape mean_A into a column vector
    covariance_A += np.dot(x, x.T) - np.dot(mean_A, mean_A.T)

covariance_A /= N_A

print("Covariance matrix for class A:")
print(covariance_A)