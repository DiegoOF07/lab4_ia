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

class Perceptron:
    def __init__(self, lr=0.01, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None

    def activation(self, z):
        return np.where(z >= 0, 1, -1)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.w = np.random.randn(n_features)
        self.b = np.random.randn()

        for _ in range(self.epochs):
            for i in range(n_samples):
                linear_output = np.dot(X[i], self.w) + self.b
                y_pred = self.activation(linear_output)

                if y_pred != y[i]:
                    self.w += self.lr * y[i] * X[i]
                    self.b += self.lr * y[i]

    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return self.activation(linear_output)
