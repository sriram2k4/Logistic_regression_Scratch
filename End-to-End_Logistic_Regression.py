import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Sigmoid Function

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Gradient Descent
def gradient_descent(X, y, theta, alpha, interations):
    '''
    :param X: Feature matrix (m, n + 1)
    :param y: Corresponidng label to Feature vector (m, 1)
    :param theta: Feature weight vector (m+1, 1)
    :param alpha: Learning rate
    :param interations: Number of iterations

    Returns:
        :param theta: Optimized feature vector 
    '''

    m = len(y)

    for i in range(interations):

        # Getting Z 
        print("There")


