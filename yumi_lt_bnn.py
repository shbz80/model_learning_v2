import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from model_leraning_utils import UGP, SimplePolicy, yumi_joint_limits
import time
import pickle
import os
import math

import dyn_pred
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from dyn_pred.dynamics_predictor import DynamicsPredictor
from dyn_pred.bnn_dyn.bnn_dyn import BNNDynamics
import tensorflow as tf
from model_leraning_utils import traj_with_globalgp

#data and control things
# logfile = "/home/shahbaz/Research/Software/model_learning/Results/yumi_peg_exp_new_preprocessed_data_train_4.p" # small data
# logfile = "/home/shahbaz/Research/Software/model_learning/Results/yumi_peg_exp_new_preprocessed_data_train_big_data_wom10.p" # small data
# logfile = "/home/shahbaz/Research/Software/model_learning/Results/yumi_peg_exp_new_preprocessed_data_train_big_data_wom10.p" # small data
# logfile = "/home/shahbaz/Research/Software/model_learning/Results/Final/yumi_peg_exp_new_preprocessed_data_train_big_data_wom10_fixed.p"
logfile = "/home/shahbaz/Research/Software/model_learning/Results/Final/yumi_peg_exp_new_preprocessed_data_train_small_data_wom10_fixed.p"
# result_file = "/home/shahbaz/Research/Software/model_learning/Results/results_yumi_bnn_bigdata.p"
# result_file = "/home/shahbaz/Research/Software/model_learning/Results/Final/results_yumi_bnn_d40.p"
result_file = "/home/shahbaz/Research/Software/model_learning/Results/Final/results_yumi_bnn_d15.p"
# result_file = "/home/shahbaz/Research/Software/model_learning/Results/results_yumi_bnn_smalldata_unstable.p"
# result_file = "/home/shahbaz/Research/Software/model_learning/Results/results_yumi_bnn_smalldata.p"

exp_data = pickle.load( open(logfile, "rb" ), encoding='latin1' )

# bnn_results = {}
bnn_results = pickle.load( open(result_file, "rb" ), encoding='latin1' )
# bnn_results['rmse'] = []
# bnn_results['nll'] = []

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

XUs_t_train = exp_data['XUs_t_train'][:15] # shape: n_train, T, dXU, state-action, sequential data
Xs_t1_train = exp_data['Xs_t1_train'][:15]
Xs_t_train = exp_data['Xs_t_train'][:15]

# XUs_t_train = exp_data['XUs_t_train']
# Xs_t1_train = exp_data['Xs_t1_train']
# Xs_t_train = exp_data['Xs_t_train']

XU_t_train = XUs_t_train.reshape(-1, XUs_t_train.shape[-1]) # shape: n_train*T, dXU, state-action, sequential data
X_t1_train = Xs_t1_train.reshape(-1, Xs_t1_train.shape[-1]) # shape: n_train*T, dX, next state, sequential data
X_t_train = Xs_t_train.reshape(-1, Xs_t_train.shape[-1])
U0_t_train = XUs_t_train[0, :, dX:dX+dU]
dX_t_train = X_t1_train - X_t_train
# change bnn_dyn.py to use this
# dX_t_train_max = np.max(dX_t_train, axis=0)*3
# dX_t_train_min = np.min(dX_t_train, axis=0)*3
# max_delta_range = (dX_t_train_min, dX_t_train_max)

yumi_joint_limits = np.array(yumi_joint_limits)
yumi_joint_max = yumi_joint_limits[:,1]
yumi_joint_min = yumi_joint_limits[:,0]
max_state_range = (yumi_joint_min, yumi_joint_max)

Xrs_t_train = exp_data['Xrs_t_train']
Xrs_t_test = exp_data['Xrs_t_test']
Us_t_train = exp_data['Us_t_train']
Us_t_test = exp_data['Us_t_test']
Xs_t1_test = exp_data['Xs_t1_test']
X_t1_test = Xs_t1_test.reshape(-1, Xs_t1_test.shape[-1])
Xs_t_test = exp_data['Xs_t_test']
X_t_test = Xs_t_test.reshape(-1, Xs_t_test.shape[-1])
XUs_t_test = exp_data['XUs_t_test']
XU_t_test = XUs_t_test.reshape(-1, XUs_t_test.shape[-1])
dX_t_test = X_t1_test - X_t_test

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
Kp = exp_params_rob['Kp']
pol_per_facor = -0.1
exp_params_rob['Kp'] = Kp + Kp * pol_per_facor

jitter_var_tl = 1e-6
print('dX:{0}, dU:{1}'.format(dX, dU))
delta_model = True
update_bnn_data = False
update_bnn_score = False
H = T  # prediction horizon
num_traj_samples = 50

# original simple policy
Xrs_data = Xrs_t_test
Us_data = Us_t_test
sim_pol = SimplePolicy(Xrs_data, Us_data, exp_params_rob)
pol = sim_pol.predict
# global gp long-term prediction
ugp_global_dyn = UGP(dX + dU, **ugp_params) # initialize unscented transform for dynamics
ugp_global_pol = UGP(dX, **ugp_params) # initialize unscented transform for policy
x_mu_t = exp_data['X0_mu']  # mean of initial state distr
x_var_t = np.diag(exp_data['X0_var'])
x_var_t[1,1] = 1e-6   # setting small variance for initial vel
traj_with_BNN = traj_with_globalgp(x_mu_t, x_var_t, None, None, dlt_mdl=delta_model)
dpgmm_params = {
    'n_components': 2,  # cluster size
    'covariance_type': 'full',
    'tol': 1e-6,
    'n_init': 3,
    'max_iter': 300,
    'weight_concentration_prior_type': 'dirichlet_process',
    'weight_concentration_prior': 1e-2,
    'mean_precision_prior': None,
    'mean_prior': None,
    'degrees_of_freedom_prior': 14 + 2,
    'covariance_prior': None,
    'warm_start': False,
    'init_params': 'random',
}
layer_grid = [  [dX+dU, 16, 16, 16, dX],
                [dX+dU, 32, 32, 32, dX],
                [dX+dU, 64, 64, 64, dX],
                [dX+dU, 128, 128, 128, dX],
                [dX+dU, 256, 256, 256, dX]]
# layer_grid = [  [dX+dU, 16, dX],
#                 [dX+dU, 32, dX],
#                 [dX+dU, 64, dX],
#                 [dX+dU, 128, dX],
#                 [dX+dU, 256, dX],
#                 [dX+dU, 16, 16, dX],
#                 [dX+dU, 32, 32, dX],
#                 [dX+dU, 64, 64, dX],
#                 [dX+dU, 128, 128, dX],
#                 [dX+dU, 256, 256, dX],
#                 [dX+dU, 8, 8, 8, dX],
#                 [dX+dU, 16, 16, 16, dX],
#                 [dX+dU, 32, 32, 32, dX],
#                 [dX+dU, 64, 64, 64, dX],
#                 [dX+dU, 128, 128, 128, dX],
#                 [dX+dU, 256, 256, 256, dX]]
network_layer = [dX+dU, 64, 64, 64, dX]
# n_repeat = len(layer_grid)
n_repeat = 1
h =99
grid_result = []
for network_layer in [network_layer]:
    print('Layer', network_layer)
    train_error = []
    # bnn_results['rmse'] = []
    # bnn_results['nll'] = []
    for r in range(n_repeat):
        # for BNN
        tf.reset_default_graph()
        # model = DynamicsPredictor(model=BNNDynamics(layers=[dX+dU, 128, 128, 128, dX], n_nets=20, name='bnn2d'))  # input 3d , output 2d
        model = DynamicsPredictor(
            model=BNNDynamics(layers=network_layer, n_nets=5, name='bnn2d'))  # input 3d , output 2d

        start_time = time.time()
        if delta_model:
            model.train(XU_t_train, dX_t_train)
        else:
            model.train(XU_t_train, X_t1_train)
        train_time = time.time() - start_time
        bnn_results['train_time'] = train_time

        # dX_t_test_pred, _ = model.model.bnn_model.predict(XU_t_test, factored=True)
        # bnn_test_score = np.linalg.norm(dX_t_test - dX_t_test_pred)
        # train_error.append(bnn_test_score)

        x_mu_t = exp_data['X0_mu']  # mean of initial state distr
        x_var_t = np.diag(exp_data['X0_var'])
        start_time = time.time()
        # traj_samples = model.sample_traj(x_mu_t, x_var_t, H=h, pol=pol, U=None, n_particles=num_traj_samples,
        #                                  delta_model=delta_model, unfactored=True, max_range=max_delta_range)
        # traj_samples = model.sample_traj(x_mu_t, x_var_t, H=h, pol=pol, U=None, n_particles=num_traj_samples, # small data
        #                                  delta_model=delta_model, unfactored=True, max_range=max_delta_range)
        traj_samples = model.sample_traj(x_mu_t, x_var_t, H=h, pol=pol, U=None, n_particles=num_traj_samples,
                                         # small data
                                         delta_model=delta_model, unfactored=True, max_range=max_state_range)
        # traj_samples[(traj_samples>1e100)] = 1e100
        # traj_samples = model.sample_traj(x_mu_t, x_var_t, H=h, pol=pol, U=None, n_particles=num_traj_samples, # big data
        #                                  delta_model=delta_model, unfactored=True, max_range=None)
        # traj_samples = model.sample_traj(x_mu_t, x_var_t, H=h, pol=None, U=Us_t_train[0], n_particles=num_traj_samples,
        #                                  delta_model=delta_model)
        # traj_with_BNN.sample_trajs = traj_samples
        # _, _, _, _ = traj_with_BNN.estimate_gmm_traj_density(dpgmm_params, Xs_t_test,
        #                                                                                  plot=False)
        # pred_time = time.time() - start_time
        # print('BNN pred time with density est', pred_time)

        traj_mean = np.mean(traj_samples, axis=0)
        traj_std = np.sqrt(np.var(traj_samples, axis=0))
        traj_covar = np.zeros((h, dX, dX))
        for t in range(h):
            traj_covar[t] = np.cov(traj_samples[:, t, :], rowvar=False)
        pred_time = time.time() - start_time
        bnn_results['pred_time'] = pred_time
        tm = np.array(range(h)) * dt
        # tm = np.array(range(h))
        plt.figure()
        plt.title('Long-term prediction with BNN')
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
            plt.fill_between(tm, traj_mean[:h, j] - traj_std[:h, j] * 1.96, traj_mean[:h, j] + traj_std[:h, j] * 1.96,
                             alpha=0.2, color='g')

        # jVel
        for j in range(dV):
            plt.subplot(3, 7, 8 + j)
            # plt.xlabel('Time (s)')
            # plt.ylabel('Joint Vel (rad/s)')
            plt.title('j%dVel' % (j + 1))
            plt.autoscale(True)
            plt.plot(tm, Xs_t_train[:, :h, dP + j].T, alpha=0.2, color='k')
            # plt.autoscale(False)
            # plt.plot(tm, traj_samples[:, :h, dP+j].T, alpha=0.2, color='b')
            plt.plot(tm, traj_mean[:h, dP + j], color='b')
            plt.fill_between(tm, traj_mean[:h, dP + j] - traj_std[:h, dP + j] * 1.96,
                             traj_mean[:h, dP + j] + traj_std[:h, dP + j] * 1.96,
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
                # x_var_t = traj_covar[t] + np.eye(dX)*jitter_var_tl
                x_var_t = traj_covar[t]
                x_var_t = np.diag(np.diag(x_var_t))
                X_test_log_ll[t, i] = sp.stats.multivariate_normal.logpdf(x_t, x_mu_t, x_var_t)
                X_test_SE[t, i] = np.dot((x_t - x_mu_t), (x_t - x_mu_t))

        nll_mean = np.mean(X_test_log_ll.reshape(-1))
        nll_std = np.std(X_test_log_ll.reshape(-1))
        rmse = np.sqrt(np.mean(X_test_SE.reshape(-1)))
        print('Yumi exp BNN', 'NLL mean: ', nll_mean, 'NLL std: ', nll_std, 'RMSE:', rmse)
        del model

        if update_bnn_score:
            bnn_results['rmse'].append(rmse)
            bnn_results['nll'].append((nll_mean, nll_std))
            pickle.dump(bnn_results, open(result_file, "wb"), protocol=2)
        if update_bnn_data:
            bnn_results['traj_samples'] = traj_samples
            pickle.dump(bnn_results, open(result_file, "wb"), protocol=2)

    # grid_result.append([network_layer, np.mean(train_error)])
    # print('rmse error', network_layer, np.mean(bnn_results['rmse']))
    # print('nll error', network_layer, np.mean(np.array(bnn_results['nll'])[:,0]))


None