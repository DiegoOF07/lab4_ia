import numpy as np

def main():
    np.random.seed(2026)

    n = 1000
    noise = 10.0

    x = np.linspace(-3, 3, n).reshape(-1, 1)
    f_x = 2*x**3 - 3*x**2 + 5*x + 3

    noise_values = np.random.normal(0, noise, n)
    y = f_x + noise_values

    x_mean, x_std = x.mean(), x.std()
    y_mean, y_std = y.mean(), y.std()

    x_norm = (x - x_mean) / x_std
    y_norm = (y - y_mean) / y_std

    polynom = np.hstack([np.ones_like(x_norm), x_norm, x_norm**2, x_norm**3])

    X_poly = np.hstack([np.ones_like(x_norm), x_norm, x_norm**2, x_norm**3])
    print(X_poly)

if __name__ == "__main__":
    main()
