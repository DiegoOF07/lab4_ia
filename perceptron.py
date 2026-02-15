import numpy as np
from sklearn.datasets import load_iris

def load_iris_binary():
    iris = load_iris()
    X = iris.data
    y = iris.target

    mask = y < 2
    X = X[mask]
    y = y[mask]

    X = X[:, :2]

    y = np.where(y == 0, -1, 1)

    return X, y