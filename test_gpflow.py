import gpflow
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('/home/shahbaz/Research/Software/GPflow/notebooks/basics/data/regression_1D.csv', delimiter=',')
X = data[:, 0].reshape(-1, 1)
Y = data[:, 1].reshape(-1, 1)

plt.plot(X, Y, 'kx', mew=2)

k = gpflow.kernels.Matern52(input_dim=1)
m = gpflow.models.GPR(X, Y, kern=k, mean_function=None)
print(m)
m.likelihood.variance = 0.01
m.kern.lengthscales = 0.3
opt = gpflow.train.ScipyOptimizer()
opt.minimize(m)

## generate test points for prediction
xx = np.linspace(-0.1, 1.1, 100).reshape(100, 1)  # test points must be of shape (N, D)

## predict mean and variance of latent GP at test points
mean, var = m.predict_f(xx)

## generate 10 samples from posterior
samples = m.predict_f_samples(xx, 10)  # shape (10, 100, 1)

## plot
plt.figure(figsize=(12, 6))
plt.plot(X, Y, 'kx', mew=2)
plt.plot(xx, mean, 'C0', lw=2)
plt.fill_between(xx[:,0],
                 mean[:,0] - 1.96 * np.sqrt(var[:,0]),
                 mean[:,0] + 1.96 * np.sqrt(var[:,0]),
                 color='C0', alpha=0.2)

plt.plot(xx, samples[:, :, 0].T, 'C0', linewidth=.5)
plt.xlim(-0.1, 1.1)

plt.show()