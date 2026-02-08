# type: ignore

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel

# Generate synthetic training data
X_train = np.linspace(0, 1, 10).reshape(-1, 1)
y_train = np.sin(2 * np.pi * X_train).ravel() + np.random.normal(0, 0.1, X_train.shape[0])

# Define kernel: constant * RBF + Matern (as an example)
kernel = ConstantKernel(1.0) * RBF(length_scale=0.2) + Matern(length_scale=0.2, nu=1.5)

# Instantiate GP regressor
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1**2, n_restarts_optimizer=10)

# Fit to data
gp.fit(X_train, y_train)

# Predict at new points
X_test = np.linspace(0, 1, 100).reshape(-1, 1)
y_pred, y_std = gp.predict(X_test, return_std=True)

# Plot results (optional)
import matplotlib.pyplot as plt
plt.figure()
plt.plot(X_train, y_train, 'ro', label='Observed')
plt.plot(X_test, y_pred, 'b-', label='GP Mean')
plt.fill_between(X_test.ravel(), y_pred - 2*y_std, y_pred + 2*y_std, color='blue', alpha=0.2, label='95% CI')
plt.legend()
plt.savefig("gp_sklearn_example.pdf")
plt.show()