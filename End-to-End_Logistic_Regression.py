import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def sigmoid(z):
    h = 1 / (1 + np.exp(-z))
    return h

def gradientDescent(x, y, theta, alpha, num_iters):
    m = len(y)
    for i in range(0,num_iters):
        z = np.dot(x, theta)
        h = sigmoid(z)
        # print(z)
        # print(h)
        theta = theta - (alpha / m) * (np.dot(x.T, (h - y)))
    return theta

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

theta = gradientDescent(X_train, Y_train, np.zeros((30)), 0.001, 700)

# print(theta)

# Predicting the test

y_pred = np.dot(X_test, theta)
y_pred = np.where(y_pred > 0.5, 1, 0)

# for i in y_pred:
#     if i > 0.5:
#         y_hat.append(1)
#     else:
#         y_hat.append(0)

print(Y_test)
print(y_pred)

accuracy = np.sum(y_pred == Y_test) / len(Y_test)
print("Accuracy: ", accuracy*100)