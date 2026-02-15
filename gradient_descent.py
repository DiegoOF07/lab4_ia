import numpy as np
import time

def batch_gradient_descent(X, y, lr=0.1, epochs=100):
    n_samples, n_features = X.shape
    w = np.random.randn(n_features, 1) * 0.01
    
    history = []
    start_time = time.time()
    
    for _ in range(epochs):
        y_hat = X @ w
        
        error = y_hat - y
        
        gradient = (1 / n_samples) * (X.T @ error)
        
        w = w - lr * gradient
        
        mse = np.mean(error**2)
        elapsed = time.time() - start_time
        history.append([elapsed, mse])
        
    return w, np.array(history)

def stochastic_gradient_descent(X, y, lr=0.01, epochs=100):
    n_samples, n_features = X.shape
    w = np.random.randn(n_features, 1) * 0.01
    history = []
    start_time = time.time()

    for _ in range(epochs):
        indices = np.random.permutation(n_samples)
        for idx in indices:
            xi = X[idx:idx+1]
            yi = y[idx:idx+1]
            
            prediction = xi @ w
            error = prediction - yi
            gradient = xi.T @ error
            
            w = w - lr * gradient
            
            mse = np.mean((X @ w - y)**2)
            history.append([time.time() - start_time, mse])
            
    return w, np.array(history)

def mini_batch_gradient_descent(X, y, lr=0.05, epochs=100, batch_size=32):
    n_samples, n_features = X.shape
    w = np.random.randn(n_features, 1) * 0.01
    history = []
    start_time = time.time()

    for _ in range(epochs):
        indices = np.random.permutation(n_samples)
        X_sh = X[indices]
        y_sh = y[indices]
        
        for i in range(0, n_samples, batch_size):
            Xi = X_sh[i : i + batch_size]
            yi = y_sh[i : i + batch_size]
            
            prediction = Xi @ w
            error = prediction - yi
            gradient = (1 / len(yi)) * (Xi.T @ error)
            
            w = w - lr * gradient
            
            mse = np.mean((X @ w - y)**2)
            history.append([time.time() - start_time, mse])
            
    return w, np.array(history)
