import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from model_leraning_utils import UGP, SimplePolicy
import time
import pickle
import os

import dyn_pred
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from dyn_pred.dynamics_predictor import DynamicsPredictor
from dyn_pred.gp_dyn.gp_dyn import GaussianProcessDynamics
from dyn_pred.gp_dyn.manifold_gp.gpflow_nn_ker import NNFeaturedSE
import gpflow
import tensorflow as tf

#exp settings
n_repeat = 10 #number of training instances for nn approaches
horizons = [99]

#data and control things
logfile = "/home/shahbaz/Research/Software/model_learning/Results/yumi_peg_exp_new_preprocessed_data_train_4.p"
result_file = "/home/shahbaz/Research/Software/model_learning/Results/results_yumi_mgp.p"

exp_data = pickle.load( open(logfile, "rb" ), encoding='latin1' )
#print(exp_data.keys())
exp_params = exp_data['exp_params']
dP = exp_params['dP'] # pos dim
dV = exp_params['dV'] # vel dim
dU = exp_params['dU'] # act dim
dX = dP+dV # state dim
T = exp_params['T'] - 1  # total time steps
dt = exp_params['dt'] # sampling time
n_train = exp_data['n_train'] # number or trials in training data
n_test = exp_data['n_test'] # number or trials in testing data

XUs_t_train = exp_data['XUs_t_train'] # shape: n_train, T, dXU, state-action, sequential data
XU_t_train = XUs_t_train.reshape(-1, XUs_t_train.shape[-1]) # shape: n_train*T, dXU, state-action, sequential data
Xs_t1_train = exp_data['Xs_t1_train']
X_t1_train = Xs_t1_train.reshape(-1, Xs_t1_train.shape[-1]) # shape: n_train*T, dX, next state, sequential data
Xs_t_train = exp_data['Xs_t_train']
X_t_train = Xs_t_train.reshape(-1, Xs_t_train.shape[-1])
U0_t_train = XUs_t_train[0, :, dX:dX+dU]
dX_t_train = X_t1_train - X_t_train
dX_t_train_max = np.max(dX_t_train, axis=0)*3
dX_t_train_min = np.min(dX_t_train, axis=0)*3
max_delta_range = (dX_t_train_min, dX_t_train_max)
Xrs_t_train = exp_data['Xrs_t_train']
Us_t_train = exp_data['Us_t_train']

ugp_params = {
    'alpha': 1.,
    'kappa': 2.,
    'beta': 0.,
}

# this is copied from mjc_exp_policy because cannot be imported into Python 3 TODO
exp_params_rob = {
            'dP': 7,
            'dV': 7,
            'dU': 7,
            'Kp': np.array([.15, .15, .12, .075, .05, .05, .05]),
            'Kd': np.array([.15, .15, .12, .075, .05, .05, .05])*10.0,
}
jitter_var_tl = 1e-6
print('dX:{0}, dU:{1}'.format(dX, dU))
delta_model = True
update_mgp_data = True
update_mgp_score = True
H = T  # prediction horizon
num_traj_samples = 100

# original simple policy
Xrs_data = Xrs_t_train
Us_data = Us_t_train
sim_pol = SimplePolicy(Xrs_data, Us_data, exp_params_rob)
pol = sim_pol.predict
# global gp long-term prediction
ugp_global_dyn = UGP(dX + dU, **ugp_params) # initialize unscented transform for dynamics
ugp_global_pol = UGP(dX, **ugp_params) # initialize unscented transform for policy

mgp_res = dict()
for h in horizons:
    mgp_res['horizon_{0}'.format(h)] = []
    mgp_results = {}
    mgp_results['rmse'] = []
    mgp_results['nll'] = []
    # mgp_results = pickle.load(open(result_file, "rb"), encoding='latin1')
    for r in range(n_repeat):
        print('Training for the {0}-th repetition...'.format(r))
        curr_res = dict()

        # for mGP
        # tf.reset_default_graph()
        model = DynamicsPredictor(model=GaussianProcessDynamics(kern=NNFeaturedSE(dX+dU, [dX+dU, 128, 128, 128, dX])))

        if delta_model:
            model.train(XU_t_train, dX_t_train)
        else:
            model.train(XU_t_train, X_t1_train)

        # prediction
        x_mu_t = exp_data['X0_mu']  # mean of initial state distr
        x_var_t = np.diag(exp_data['X0_var'])
        # x_var_t[1, 1] = 1e-6  # setting small variance for initial vel    # TODO: cholesky failing for zero v0 variance

        traj_samples = np.zeros((num_traj_samples, h, dX))
        for s in range(num_traj_samples):
            traj_samples[s][0] = np.random.multivariate_normal(x_mu_t, x_var_t)
            for t in range(1, h):
                x = traj_samples[s][t - 1]
                x = x.reshape(-1)
                um, ustd = pol(x.reshape(1, -1), t)
                u = um + np.random.randn(*(um.shape)) * ustd
                u = u.reshape(-1)
                xu = np.append(x, u)
                mu, var = model.model.predict_f(xu.reshape(1, -1))
                mu = mu.reshape(-1)
                var = var.reshape(-1)
                var = np.diag(var)
                if not delta_model:
                    x_new = np.random.multivariate_normal(mu, var)
                else:
                    dx_new = np.random.multivariate_normal(mu, var)
                    dx_new[dx_new > max_delta_range[1]] = 0.
                    dx_new[dx_new < max_delta_range[0]] = 0.
                    x_new = dx_new + traj_samples[s][t - 1]
                traj_samples[s][t] = x_new

        traj_mean = np.mean(traj_samples, axis=0)
        traj_std = np.sqrt(np.var(traj_samples, axis=0))
        traj_covar = np.zeros((h, dX, dX))
        for t in range(h):
            traj_covar[t] = np.cov(traj_samples[:, t, :], rowvar=False)

            # tm = np.array(range(h)) * dt
        tm = np.array(range(h))
        plt.figure()
        plt.title('Long-term prediction with mGP')
        # jPos
        for j in range(dP):
            plt.subplot(3, 7, 1 + j)
            # plt.xlabel('Time (s)')
            # plt.ylabel('Joint Pos (rad)')
            plt.title('j%dPos' % (j + 1))
            plt.autoscale(True)
            plt.plot(tm, Xs_t_train[:, :h, j].T, alpha=0.2, color='k')
            # plt.autoscale(False)
            # plt.plot(tm, traj_samples[:, :h, j].T, alpha=0.2, color='g')
            plt.plot(tm, traj_mean[:h, j], color='g')
            plt.fill_between(tm, traj_mean[:h, j] - traj_std[:h, j] * 1.96, traj_mean[:h, j] + traj_std[:h, j] * 1.96, alpha=0.2, color='g')

        # jVel
        for j in range(dV):
            plt.subplot(3, 7, 8 + j)
            # plt.xlabel('Time (s)')
            # plt.ylabel('Joint Vel (rad/s)')
            plt.title('j%dVel' % (j + 1))
            plt.autoscale(True)
            plt.plot(tm, Xs_t_train[:, :h, dP + j].T, alpha=0.2, color='k')
            # plt.autoscale(False)
            # plt.plot(tm, traj_samples[:, :h, dP + j].T, alpha=0.2, color='b')
            plt.plot(tm, traj_mean[:h, dP+j], color='b')
            plt.fill_between(tm, traj_mean[:h, dP+j] - traj_std[:h, dP+j] * 1.96, traj_mean[:h, dP+j] + traj_std[:h, dP+j] * 1.96,
                             alpha=0.2, color='b')

        for j in range(dV):
            plt.subplot(3, 7, 15 + j)
            # plt.xlabel('Time (s)')
            # plt.ylabel('Joint Trq (Nm)')
            plt.title('j%dTrq' % (j + 1))
            plt.autoscale(True)
            plt.plot(tm, XUs_t_train[:, :h, dX + j].T, alpha=0.2, color='k')
        plt.legend()
        plt.show(block=False)



        # loglikelihood score
        XUs_t_test = exp_data['XUs_t_test']
        assert (XUs_t_test.shape[0] == n_test)
        X_test_log_ll = np.zeros((h, n_test))
        X_test_SE = np.zeros((h, n_test))
        for t in range(h):  # one data point less than in XU_test
            for i in range(n_test):
                XU_test = XUs_t_test[i]
                x_t = XU_test[t, :dX]
                x_mu_t = traj_mean[t]
                x_var_t = traj_covar[t] + np.eye(dX) * jitter_var_tl
                X_test_log_ll[t, i] = sp.stats.multivariate_normal.logpdf(x_t, x_mu_t, x_var_t)
                X_test_SE[t, i] = np.dot((x_t - x_mu_t), (x_t - x_mu_t))

        nll_mean = np.mean(X_test_log_ll.reshape(-1))
        nll_std = np.std(X_test_log_ll.reshape(-1))
        rmse = np.sqrt(np.mean(X_test_SE.reshape(-1)))
        print('Yumi exp mGP', 'NLL mean: ', nll_mean, 'NLL std: ', nll_std, 'RMSE:', rmse)

        curr_res['ll_score'] = X_test_log_ll

        mgp_res['horizon_{0}'.format(h)].append(curr_res)

        del model
        # mgp_results = {}
        # mgp_results['rmse'] = []
        # mgp_results['nll'] = []
        # mgp_results = pickle.load(open(result_file, "rb"), encoding='latin1')
        if update_mgp_score:
            mgp_results['rmse'].append(rmse)
            mgp_results['nll'].append((nll_mean, nll_std))
            # pickle.dump(mgp_results, open(result_file, "wb"), protocol=2)
        if update_mgp_data:
            mgp_results['traj_samples'] = traj_samples
            # pickle.dump(mgp_results, open(result_file, "wb"), protocol=2)
    pickle.dump(mgp_results, open(result_file, "wb"), protocol=2)
np.save('mgp_res', mgp_res)

#visualize results
mgp_res = np.load('mgp_res.npy').item()['horizon_99']

#lets see scores
eval_horizons = [4, 49, 99]
approaches = ['mGP']

mgp_score_avg = []
mgp_score_std = []

for h in eval_horizons:
    mgp_score = []
    for trial in range(1):
        mgp_score = mgp_res[trial]['ll_score'][:h]
    mgp_score_avg.append(np.mean(mgp_score))
    mgp_score_std.append(np.std(mgp_score))

print(mgp_score_avg, mgp_score_std)
#%matplotlib auto
fig = plt.figure()
ax = fig.add_subplot(111)
mgp_plt = ax.bar(eval_horizons, mgp_score_avg, yerr=mgp_score_std)
ax.set_xlabel('Prediction Horizon', fontsize=16)
ax.set_ylabel('Log-likelihood Score', fontsize=16)
ax.legend((mgp_plt), approaches, fontsize=16)
ax.grid()
plt.show()