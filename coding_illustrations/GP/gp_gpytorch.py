# type: ignore

import torch
import gpytorch
import numpy as np

# Generate synthetic data
X_train = torch.linspace(0, 1, 10).unsqueeze(-1)
y_train = torch.sin(2 * np.pi * X_train).squeeze() + 0.1 * torch.randn(X_train.size(0))

# Define GP model
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Instantiate likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPModel(X_train, y_train, likelihood)

# Training loop
model.train()
likelihood.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(100):
    optimizer.zero_grad()
    output = model(X_train)
    loss = -mll(output, y_train)
    loss.backward()
    optimizer.step()

# Switch to eval mode for prediction
model.eval()
likelihood.eval()
X_test = torch.linspace(0, 1, 100).unsqueeze(-1)
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    pred = model(X_test)
    y_pred = pred.mean
    y_std = pred.stddev

# Plot results (optional)
import matplotlib.pyplot as plt
plt.figure()
plt.plot(X_train.numpy(), y_train.numpy(), 'ro', label='Observed')
plt.plot(X_test.numpy(), y_pred.numpy(), 'b-', label='GP Mean')
plt.fill_between(X_test.numpy().ravel(),
                 (y_pred - 2*y_std).numpy(),
                 (y_pred + 2*y_std).numpy(),
                 color='blue', alpha=0.2, label='95% CI')
plt.legend()
plt.savefig("gp_gpytorch_example.pdf")
plt.show()