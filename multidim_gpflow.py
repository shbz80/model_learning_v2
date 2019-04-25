from __future__ import print_function
import numpy as np
import time
import gpflow


class MdGpflowGP(object):
    def __init__(self, gpr_params, out_dim):
        self.gp_param = gpr_params
        self.out_dim = out_dim

    def fit(self, X, Y):
        assert(Y.shape[1]>=2)
        assert (X.shape[1]>=2)
        self.gp_list = []
        in_dim = X.shape[1]
        for i in range(self.out_dim):
            gp_params = self.gp_param
            normalize = gp_params['normalize']


            kernel = gpflow.kernels.RBF(input_dim=in_dim, ARD=True)
            y = Y[:,i].reshape(-1,1)
            m = gpflow.models.GPR(X, y, kern=kernel)

            x_sig = np.sqrt(np.var(X, axis=0))
            len_scale = x_sig
            len_scale_lb = np.min(x_sig/10.)
            len_scale_ub = np.max(x_sig / 1.)
            len_scale_b = (len_scale_lb, len_scale_ub)
            noise_var = gp_params['noise_var'][i] #1e-3
            y_var = np.var(Y[:,i])
            sig_var = y_var
            # sig_var = y_var - noise_var
            sig_var_b = (sig_var/10., sig_var*10.)

            # m.rbf.lengthscale[:] = len_scale
            m.kern.lengthscales = len_scale
            # m.rbf.lengthscale.constrain_bounded(len_scale_b[0], len_scale_b[1])
            # m.rbf.variance[:] = sig_var
            m.kern.variance = sig_var
            # m.rbf.variance.fix()
            # m.rbf.variance.constrain_bounded(sig_var_b[0], sig_var_b[1])
            # m.Gaussian_noise[:] = noise_var
            m.likelihood.variance = noise_var
            m.likelihood.variance.trainable = False
            # m.Gaussian_noise.fix()
            # m.Gaussian_noise.constrain_bounded(noise_var_b[0], noise_var_b[1])
            start_time = time.time()
            # m.optimize_restarts(optimizer='lbfgs', num_restarts=1)
            opt = gpflow.train.ScipyOptimizer()
            # m.optimize()
            opt.minimize(m)
            print ('GP',i, 'fit time', time.time() - start_time)
            self.gp_list.append(m)

    def predict(self, X, return_std=True):
        Y_mu = np.zeros((X.shape[0], self.out_dim))
        Y_std = np.zeros((X.shape[0], self.out_dim))
        for i in range(self.out_dim):
            gp = self.gp_list[i]
            # mu, var = gp.predict_noiseless(X)
            mu, var = gp.predict_f(X)
            # mu, var = gp.predict(X)
            Y_mu[:, i] = mu.reshape(-1)
            Y_std[:, i] = np.sqrt(var).reshape(-1)
        return Y_mu, Y_std