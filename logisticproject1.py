import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

iris = load_iris(as_frame=True)

# Data preprocessing (Any method)
X = iris.data[["petal width (cm)"]]
y = iris.target_names[iris.target] == 'virginica'

# print(type(X))
# print(type(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print(X_train)
# print(y_train)
# print(X_test)
# print(y_test)

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

# y_pred = log_reg.predict(X_test)

# print(y_test)
# print(y_pred)

X_new = np.linspace(0, 3, 1000).reshape(-1, 1) # reshape to get a column vector

# Classifying flowers using pedal width

# y_proba = log_reg.predict_proba(X_new)
# decision_boundary = X_new[y_proba[:, 1] >= 0.5][0, 0]
# plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2,label="Not Iris virginica proba")
# plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris virginica proba")
# plt.plot([decision_boundary, decision_boundary], [0, 1], "k:", linewidth=2,label="Decision boundary")
# plt.show()