import numpy as np
import matplotlib.pyplot as plt

from gradient_descent import batch_gradient_descent, mini_batch_gradient_descent, stochastic_gradient_descent

from perceptron import load_iris_binary, Perceptron
import numpy as np

def main():
    np.random.seed(2026)
    n = 1000
    noise = 10.0

    x = np.linspace(-3, 3, n).reshape(-1, 1)
    f_x = 2*x**3 - 3*x**2 + 5*x + 3
    noise_values = np.random.normal(0, noise, n).reshape(-1, 1)
    y = f_x + noise_values

    x_norm = (x - x.mean()) / x.std()
    y_norm = (y - y.mean()) / y.std()

    polynomial = np.hstack([np.ones_like(x_norm), x_norm, x_norm**2, x_norm**3])

    epochs = 50

    _, hist_batch = batch_gradient_descent(polynomial, y_norm, lr=0.1, epochs=epochs)
    _, hist_sgd = stochastic_gradient_descent(polynomial, y_norm, lr=0.005, epochs=epochs)
    _, hist_mini = mini_batch_gradient_descent(polynomial, y_norm, lr=0.05, epochs=epochs, batch_size=32)

    plt.figure(figsize=(12, 6))
    plt.plot(hist_batch[:, 0], hist_batch[:, 1], label='Batch GD', linewidth=2)
    plt.plot(hist_sgd[:, 0], hist_sgd[:, 1], label='SGD', alpha=0.5)
    plt.plot(hist_mini[:, 0], hist_mini[:, 1], label='Mini-batch GD', alpha=0.8)

    plt.yscale('log')
    plt.xlabel('Tiempo Real (Segundos)')
    plt.ylabel('Costo (MSE)')
    plt.title('Comparación de Algoritmos: Estabilidad vs Velocidad')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()

    perceptron_dataset()

def perceptron_dataset():
    X, y = load_iris_binary()

    model = Perceptron(lr=0.01, epochs=100)
    model.fit(X, y)

    y_pred = model.predict(X)
    accuracy = np.mean(y_pred == y)

    print("Accuracy:", accuracy)

if __name__ == "__main__":
    main()
