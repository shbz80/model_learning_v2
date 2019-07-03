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
from dyn_pred.bnn_dyn.bnn_dyn import BNNDynamics
import tensorflow as tf
from model_leraning_utils import traj_with_globalgp

#exp settings
n_repeat = 1  #number of training instances for nn approaches
horizons = [99]

#data and control things
# logfile = "/home/shahbaz/Research/Software/model_learning/Results/yumi_peg_exp_new_preprocessed_data_train_4.p" # small data
# logfile = "/home/shahbaz/Research/Software/model_learning/Results/yumi_peg_exp_new_preprocessed_data_train_big_data_wom10.p" # small data
# logfile = "/home/shahbaz/Research/Software/model_learning/Results/yumi_peg_exp_new_preprocessed_data_train_big_data_wom10.p" # small data
logfile = "/home/shahbaz/Research/Software/model_learning/Results/Final/yumi_peg_exp_new_preprocessed_data_train_big_data_wom10_fixed.p"
# result_file = "/home/shahbaz/Research/Software/model_learning/Results/results_yumi_bnn_bigdata.p"
result_file = "/home/shahbaz/Research/Software/model_learning/Results/Final/results_yumi_bnn_d40.p"
# result_file = "/home/shahbaz/Research/Software/model_learning/Results/results_yumi_bnn_smalldata_unstable.p"
# result_file = "/home/shahbaz/Research/Software/model_learning/Results/results_yumi_bnn_smalldata.p"

exp_data = pickle.load( open(logfile, "rb" ), encoding='latin1' )

bnn_results = {}
# bnn_results = pickle.load( open(result_file, "rb" ), encoding='latin1' )
bnn_results['rmse'] = []
bnn_results['nll'] = []

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

XU_t_train = XUs_t_train.reshape(-1, XUs_t_train.shape[-1]) # shape: n_train*T, dXU, state-action, sequential data
X_t1_train = Xs_t1_train.reshape(-1, Xs_t1_train.shape[-1]) # shape: n_train*T, dX, next state, sequential data
X_t_train = Xs_t_train.reshape(-1, Xs_t_train.shape[-1])
U0_t_train = XUs_t_train[0, :, dX:dX+dU]
dX_t_train = X_t1_train - X_t_train
dX_t_train_max = np.max(dX_t_train, axis=0)*3
dX_t_train_min = np.min(dX_t_train, axis=0)*3
max_delta_range = (dX_t_train_min, dX_t_train_max)
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
layer_grid = [  [dX+dU, 16, dX],
                [dX+dU, 32, dX],
                [dX+dU, 64, dX],
                [dX+dU, 128, dX],
                [dX+dU, 256, dX],
                [dX+dU, 16, 16, dX],
                [dX+dU, 32, 32, dX],
                [dX+dU, 64, 64, dX],
                [dX+dU, 128, 128, dX],
                [dX+dU, 256, 256, dX],
                [dX+dU, 8, 8, 8, dX],
                [dX+dU, 16, 16, 16, dX],
                [dX+dU, 32, 32, 32, dX],
                [dX+dU, 64, 64, 64, dX],
                [dX+dU, 128, 128, 128, dX],
                [dX+dU, 256, 256, 256, dX]]
# layer_grid = [  [dX+dU, 32, 32, 32, dX],
#                 [dX+dU, 64, 64, 64, dX],
#                 [dX+dU, 128, 128, 128, dX],
#                 [dX+dU, 256, 256, 256, dX]]
network_layer = [dX+dU, 64, 64, 64, dX]
# n_repeat = len(layer_grid)
n_repeat = 1
bnn_res = dict()
for h in horizons:
    bnn_res['horizon_{0}'.format(h)] = []
    for r in range(n_repeat):
        print('Training for the {0}-th repetition...'.format(r))
        curr_res = dict()

        # for BNN
        tf.reset_default_graph()
        # model = DynamicsPredictor(model=BNNDynamics(layers=[dX+dU, 128, 128, 128, dX], n_nets=20, name='bnn2d'))  # input 3d , output 2d
        model = DynamicsPredictor(
            model=BNNDynamics(layers=layer_grid[r], n_nets=10, name='bnn2d'))  # input 3d , output 2d

        start_time = time.time()
        if delta_model:
            model.train(XU_t_train, dX_t_train)
        else:
            model.train(XU_t_train, X_t1_train)
        train_time = time.time() - start_time
        bnn_results['train_time'] = train_time

        dX_t_test_pred, _ = model.model.bnn_model.predict(XU_t_test, factored=False)
        bnn_test_score = np.linalg.norm(dX_t_test - dX_t_test_pred)
        print('bnn 1step test score for', layer_grid[r], bnn_test_score)

        x_mu_t = exp_data['X0_mu']  # mean of initial state distr
        x_var_t = np.diag(exp_data['X0_var'])
        start_time = time.time()
        # traj_samples = model.sample_traj(x_mu_t, x_var_t, H=h, pol=pol, U=None, n_particles=num_traj_samples,
        #                                  delta_model=delta_model, unfactored=True, max_range=max_delta_range)
        # traj_samples = model.sample_traj(x_mu_t, x_var_t, H=h, pol=pol, U=None, n_particles=num_traj_samples, # small data
        #                                  delta_model=delta_model, unfactored=True, max_range=max_delta_range)
        traj_samples = model.sample_traj(x_mu_t, x_var_t, H=h, pol=pol, U=None, n_particles=num_traj_samples,
                                         # small data
                                         delta_model=delta_model, unfactored=False, max_range=None)
        # traj_samples = model.sample_traj(x_mu_t, x_var_t, H=h, pol=pol, U=None, n_particles=num_traj_samples, # big data
        #                                  delta_model=delta_model, unfactored=True, max_range=None)
        # traj_samples = model.sample_traj(x_mu_t, x_var_t, H=h, pol=None, U=Us_t_train[0], n_particles=num_traj_samples,
        #                                  delta_model=delta_model)
        traj_with_BNN.sample_trajs = traj_samples
        _, _, _, _ = traj_with_BNN.estimate_gmm_traj_density(dpgmm_params, Xs_t_test,
                                                                                         plot=False)
        pred_time = time.time() - start_time
        print('BNN pred time with density est', pred_time)

        traj_mean = np.mean(traj_samples, axis=0)
        traj_std = np.sqrt(np.var(traj_samples, axis=0))
        traj_covar = np.zeros((h, dX, dX))
        for t in range(h):
            traj_covar[t] = np.cov(traj_samples[:, t, :], rowvar=False)
        pred_time = time.time() - start_time
        bnn_results['pred_time'] = pred_time
        # tm = np.array(range(h)) * dt
        tm = np.array(range(h))
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


        # # prediction
        # x_mu_t = exp_data['X0_mu']  # mean of initial state distr
        # x_var_t = np.diag(exp_data['X0_var'])
        # # x_var_t[1, 1] = 1e-6  # setting small variance for initial vel    # TODO: cholesky failing for zero v0 variance
        #
        # X_mu_pred = []  # list for collecting state mean
        # X_var_pred = []  # list for collecting state var
        # U_mu_pred = []
        # U_var_pred = []
        # # for our baseline models to predict
        # for t in range(h):
        #     u_mu_t, u_var_t, _, _, xu_cov = ugp_global_pol.get_posterior(sim_pol, x_mu_t, x_var_t, t)
        #     # u_mu_t = Us_t_train[0][t]
        #     # u_var_t = np.eye(dU)*1e-6
        #     U_mu_pred.append(u_mu_t)
        #     U_var_pred.append(u_var_t)
        #     X_mu_pred.append(x_mu_t)
        #     X_var_pred.append(x_var_t)
        #     # X_particles.append(Y_mu)
        #     xu_mu_t = np.append(x_mu_t, u_mu_t)
        #     xu_var_t = np.block([[x_var_t, xu_cov],
        #                          [xu_cov.T, u_var_t]])
        #     # xu_var_t = np.block([[x_var_t, np.zeros((dX,dU))],
        #     #                     [np.zeros((dU,dX)), u_var_t]])
        #
        #     if not delta_model:
        #         x_mu_t, x_var_t, Y_mu, _, _ = ugp_global_dyn.get_posterior_bnn(model.model.bnn_model, xu_mu_t, xu_var_t)
        #     else:
        #         dx_mu_t, dx_var_t, dY_mu, _, xudx_covar = ugp_global_dyn.get_posterior_bnn(model.model.bnn_model, xu_mu_t, xu_var_t)
        #         xdx_covar = xudx_covar[:dX, :]
        #         x_mu_t = X_mu_pred[t] + dx_mu_t
        #         x_var_t = X_var_pred[t] + dx_var_t + xdx_covar + xdx_covar.T
        #         # x_var_t = X_var_pred[t] + dx_var_t
        #     # x_var_t = x_var_t + np.eye(dX, dX) * jitter_var_tl  # to prevent collapse of the Gaussian
        #
        # # print(len(X_mu_pred), len(X_var_pred))
        # curr_res['pred_x_mu'] = X_mu_pred
        # curr_res['pred_x_var'] = X_var_pred
        #
        #
        # # loglikelihood score
        # XUs_t_test = exp_data['XUs_t_test']
        # assert (XUs_t_test.shape[0] == n_test)
        # X_test_log_ll = np.zeros((h, n_test))
        # for t in range(h):  # one data point less than in XU_test
        #     for i in range(n_test):
        #         XU_test = XUs_t_test[i]
        #         x_t = XU_test[t, :dX]
        #         x_mu_t = X_mu_pred[t]
        #         x_var_t = X_var_pred[t]
        #         X_test_log_ll[t, i] = sp.stats.multivariate_normal.logpdf(x_t, x_mu_t, x_var_t)
        #
        # nll_mean = np.mean(X_test_log_ll.reshape(-1))
        # nll_std = np.std(X_test_log_ll.reshape(-1))
        # print('NLL mean: ', nll_mean, 'NLL std: ', nll_std)
        #
        # # plot long-term prediction
        # X_mu_pred = np.array(X_mu_pred)
        # U_mu_pred = np.array(U_mu_pred)
        # P_mu_pred = X_mu_pred[:, :dP]
        # V_mu_pred = X_mu_pred[:, dP:]
        # P_sig_pred = np.zeros((h, dP))
        # V_sig_pred = np.zeros((h, dV))
        # U_sig_pred = np.zeros((h, dU))
        # for t in range(h):
        #     P_sig_pred[t] = np.sqrt(np.diag(X_var_pred[t])[:dP])
        #     V_sig_pred[t] = np.sqrt(np.diag(X_var_pred[t])[dP:])
        #     U_sig_pred[t] = np.sqrt(np.diag(U_var_pred[t]))
        #
        #     # tm = np.array(range(H)) * dt
        # tm = np.array(range(H))
        # plt.figure()
        # plt.title('Long-term prediction with BNN')
        # # jPos
        # for j in range(dP):
        #     plt.subplot(3, 7, 1 + j)
        #     # plt.xlabel('Time (s)')
        #     # plt.ylabel('Joint Pos (rad)')
        #     plt.title('j%dPos' % (j + 1))
        #     plt.plot(tm, Xs_t_train[:, :H, j].T, alpha=0.2)
        #     plt.plot(tm, P_mu_pred[:H, j], color='g', marker='s', markersize=2, )
        #     plt.fill_between(tm, P_mu_pred[:H, j] - P_sig_pred[:H, j] * 1.96,
        #                      P_mu_pred[:H, j] + P_sig_pred[:H, j] * 1.96, alpha=0.2, color='g')
        # # jVel
        # for j in range(dV):
        #     plt.subplot(3, 7, 8 + j)
        #     # plt.xlabel('Time (s)')
        #     # plt.ylabel('Joint Vel (rad/s)')
        #     plt.title('j%dVel' % (j + 1))
        #     plt.plot(tm, Xs_t_train[:, :H, dP + j].T, alpha=0.2)
        #     plt.plot(tm, V_mu_pred[:H, j], color='b', marker='s', markersize=2, )
        #     plt.fill_between(tm, V_mu_pred[:H, j] - V_sig_pred[:H, j] * 1.96,
        #                      V_mu_pred[:H, j] + V_sig_pred[:H, j] * 1.96,
        #                      alpha=0.2, color='b')
        # for j in range(dV):
        #     plt.subplot(3, 7, 15 + j)
        #     # plt.xlabel('Time (s)')
        #     # plt.ylabel('Joint Trq (Nm)')
        #     plt.title('j%dTrq' % (j + 1))
        #     plt.plot(tm, XUs_t_train[:, :H, dX + j].T, alpha=0.2)
        #     plt.plot(tm, U_mu_pred[:H, j], color='r', marker='s', markersize=2, label='mean pred')
        #     plt.fill_between(tm, U_mu_pred[:H, j] - U_sig_pred[:H, j] * 1.96,
        #                      U_mu_pred[:H, j] + U_sig_pred[:H, j] * 1.96,
        #                      alpha=0.2, color='r')
        # plt.legend()
        # plt.show(block=False)

        curr_res['ll_score'] = X_test_log_ll

        bnn_res['horizon_{0}'.format(h)].append(curr_res)

        del model

        if update_bnn_score:
            bnn_results['rmse'].append(rmse)
            bnn_results['nll'].append((nll_mean, nll_std))
            pickle.dump(bnn_results, open(result_file, "wb"), protocol=2)
        if update_bnn_data:
            bnn_results['traj_samples'] = traj_samples
            pickle.dump(bnn_results, open(result_file, "wb"), protocol=2)

# np.save('bnn_res', bnn_res)
#
# #visualize results
# bnn_res = np.load('bnn_res.npy').item()['horizon_99']
# #lets see scores
# eval_horizons = [4, 49, 99]
#
# approaches = ['BNN']
#
# bnn_score_avg = []
# bnn_score_std = []
#
# for h in eval_horizons:
#     bnn_score = []
#     for trial in range(1):
#         bnn_score = bnn_res[trial]['ll_score'][:h]
#     bnn_score_avg.append(np.mean(bnn_score))
#     bnn_score_std.append(np.std(bnn_score))
#
# print(bnn_score_avg, bnn_score_std)
#
# #%matplotlib auto
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# bnn_plt = ax.bar(np.array(eval_horizons)-0.75, bnn_score_avg, yerr=bnn_score_std)
# ax.set_xlabel('Prediction Horizon', fontsize=16)
# ax.set_ylabel('Log-likelihood Score', fontsize=16)
# ax.legend((bnn_plt), approaches, fontsize=16)
# ax.grid()
# plt.show()
None