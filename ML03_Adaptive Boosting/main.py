'''
*** Machine Learning Mar/2024
*** Adaptive Boosting
*** Ehab M. Iqnaibi
'''
from math import log, exp
import matplotlib.pyplot as plt
import numpy as np

# Input Data
data =[[1,5,1],[2,3,1],[3,2,-1],[4,6,-1],[4,7,1],[5,9,1],[6,5,-1],[6,7,1],[8,5,-1],[8,8,-1]]

# Define Weak Classifiers
def h1(x, y):
    if x > 2.5: return -1
    return 1
def h2(x, y):
    if x > 7: return -1
    return 1
def h3(x, y):
    if y > 6.5: return 1
    return -1

h= [h1, h2, h3]

# Classifiers Success Rate
h_sr = [sum([h[j](data[i][0], data[i][1]) == data[i][2] for i in range(len(data))]) / len(data) for j in range(len(h))]
print('Classifiers Success Rate' ,h_sr)

# Sort Classifiers by Success Rate (Best to Worst)
h_sorted = [i for _, i in sorted(zip(h_sr, h), reverse=True, key=lambda pair: pair[0])]
print('Sort Classifiers' ,h_sorted)

# AdaBoost Algorithm

D = [1 / len(data) for i in data]
#print("int D ",D)
alphas = []

for clf in h_sorted:
    epsilon = sum([D[i] if clf(data[i][0], data[i][1]) != data[i][2] else 0 for i in range(len(data))])
    print('epsilon: ',epsilon)
    alpha = 0.5 * log((1 - epsilon) / epsilon)
    print('alpha: ', alpha)
    alphas.append(alpha)
    D = [D[i] * exp(alpha if clf(data[i][0], data[i][1]) != data[i][2] else 0) for i in range(len(data))]
    # Normalize D
    D = [i / sum(D) for i in D]
    print('D ',D)

print(alphas)

def strong_clf(x, y):
    total = sum([alphas[i] * h_sorted[i](x, y) for i in range(len(h))])
    if total >= 0: return 1
    return -1

# Strong Classifier Success Rate
strong_sr = sum([strong_clf(data[i][0], data[i][1]) == data[i][2] for i in range(len(data))]) / len(data)
print("Strong Classifier Success Rate ",strong_sr)



# Generate a mesh grid of points for visualization
x_min, x_max = min([d[0] for d in data]), max([d[0] for d in data])
y_min, y_max = min([d[1] for d in data]), max([d[1] for d in data])
xx, yy = np.meshgrid(np.linspace(x_min-1, x_max+1, 100), np.linspace(y_min-1, y_max+1, 100))

# Predict the class of each point in the mesh grid using the strong classifier
Z = np.array([strong_clf(xi, yi) for xi, yi in zip(xx.ravel(), yy.ravel())])
Z = Z.reshape(xx.shape)

# Plot the decision boundary and data points
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)
for d in data:
    plt.scatter(d[0], d[1], c='red' if d[2] == -1 else 'blue', marker='o')

# Plot the weak classifiers' decision boundaries
plt.axvline(x=2.5, linestyle='--', color='gray')
plt.axvline(x=7, linestyle='--', color='gray')
plt.axhline(y=6.5, linestyle='--', color='gray')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary and Weak Classifiers')
plt.show()