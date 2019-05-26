import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
from multidim_gp import MultidimGP
from model_leraning_utils import UGP
import time
import pickle
from blocks_sim import MassSlideWorld
import os

# import sys
# if not '/home/hangyin/workspace/sandbox/' in sys.path:
#     sys.path.append('/home/hangyin/workspace/sandbox/')
#     sys.path.append('/home/hangyin/workspace/sandbox/dyn_pred')
# print(sys.path)
import dyn_pred
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from dyn_pred.dynamics_predictor import DynamicsPredictor
from dyn_pred.bnn_dyn.bnn_dyn import BNNDynamics
from dyn_pred.gp_dyn.gp_dyn import GaussianProcessDynamics
from dyn_pred.gp_dyn.manifold_gp.gpflow_nn_ker import NNFeaturedSE

import gpflow
import tensorflow as tf

#exp settings
n_repeat = 1  #number of training instances for nn approaches
horizons = [74]

#data and control things
logfile = "/home/shahbaz/Research/Software/model_learning/Results/blocks_exp_preprocessed_data_rs_1_mm.p"

exp_data = pickle.load( open(logfile, "rb" ), encoding='latin1' )
#print(exp_data.keys())
exp_params = exp_data['exp_params']
Xg = exp_data['Xg']  # sate ground truth
Ug = exp_data['Ug']  # action ground truth
dP = exp_params['dP'] # pos dim
dV = exp_params['dV'] # vel dim
dU = exp_params['dU'] # act dim
dX = dP+dV # state dim
T = exp_params['T'] - 1 # total time steps
dt = exp_params['dt'] # sampling time
n_train = exp_data['n_train'] # number or trials in training data
n_test = exp_data['n_test'] # number or trials in testing data

XUs_t_train = exp_data['XUs_t_train'] # shape: n_train, T, dXU, state-action, sequential data
XU_t_train = XUs_t_train.reshape(-1, XUs_t_train.shape[-1]) # shape: n_train*T, dXU, state-action, sequential data
Xs_t1_train = exp_data['Xs_t1_train']
X_t1_train = Xs_t1_train.reshape(-1, Xs_t1_train.shape[-1]) # shape: n_train*T, dX, next state, sequential data
Xs_t_train = exp_data['Xs_t_train']
U0_t_train = XUs_t_train[0, :, dX:dX+dU]

ugp_params = {
    'alpha': 1.,
    'kappa': 2.,
    'beta': 0.,
}

print('dX:{0}, dU:{1}'.format(dX, dU))

policy_params = exp_params['policy'] # TODO: the block_sim code assumes only 'm1' mode for control
expl_noise = policy_params['m1']['noise_pol']
H = T  # prediction horizon

gpr_params = {
            # 'alpha': 1e-2,  # alpha=0 when using white kernal
            'alpha': 0.,  # alpha=0 when using white kernal
            'kernel': C(1.0, (1e-2, 1e2)) * RBF(np.ones(dX + dU), (1e-2, 1e2)) + W(noise_level=1.,
                                                                                   noise_level_bounds=(1e-4, 1e1)),
            # 'kernel': C(1.0, (1e-1, 1e1)) * RBF(np.ones(dX + dU), (1e-1, 1e1)),
            'n_restarts_optimizer': 10,
            'normalize_y': False,  # is not supported in the propogation function
        }

gpr_params_list = []
gpr_params_list.append(gpr_params)
gpr_params_list.append(gpr_params)
# gpr_params_list.append(gpr_params_p_d)
# gpr_params_list.append(gpr_params_v_d)


#and world
# global gp long-term prediction
massSlideParams = exp_params['massSlide']
# policy_params = exp_params['policy']
massSlideWorld = MassSlideWorld(**massSlideParams)
massSlideWorld.set_policy(policy_params)
massSlideWorld.reset()
mode = 'm1'  # only one mode for control no matter what X

ugp_global_dyn = UGP(dX + dU, **ugp_params) # initialize unscented transform for dynamics
ugp_global_pol = UGP(dX, **ugp_params) # initialize unscented transform for policy

x_mu_t = exp_data['X0_mu']  # mean of initial state distr
x_var_t = np.diag(exp_data['X0_var'])
x_var_t[1,1] = 1e-6   # setting small variance for initial vel    # TODO: cholesky failing for zero v0 variance
X0 = np.random.multivariate_normal(x_mu_t, x_var_t, 6)

############ Policy assumptions #######
'''
X # input state
L = np.array([.2, 1.])
Xtrg =  18.
noise = 3.
dX = np.array([Xtrg, 0.]).reshape(1,2) - X
U = np.dot(dX, L) # simple linear controller
U = U.reshape(X.shape[0],1)
if return_std:
    U_noise = np.full((U.shape), np.sqrt(noise))
return U, U_noise
'''

# bnn_res = dict()
# for h in horizons:
#     bnn_res['horizon_{0}'.format(h)] = []
#     for r in range(n_repeat):
#         print('Training for the {0}-th repetition...'.format(r))
#         curr_res = dict()
#
#         # for BNN
#         tf.reset_default_graph()
#         model = DynamicsPredictor(model=BNNDynamics(layers=[3, 32, 2], n_nets=5, name='bnn2d'))  # input 3d , output 2d
#
#         model.train(XU_t_train, X_t1_train)
#
#         # prediction
#         x_mu_t = exp_data['X0_mu']  # mean of initial state distr
#         x_var_t = np.diag(exp_data['X0_var'])
#         x_var_t[1, 1] = 1e-6  # setting small variance for initial vel    # TODO: cholesky failing for zero v0 variance
#
#         X_mu_pred = []  # list for collecting state mean
#         X_var_pred = []  # list for collecting state var
#         # for our baseline models to predict
#         for t in range(h):
#             # UT method on stochastic policy, policy is deterministic controller plus exploration noise
#             u_mu_t, u_var_t, _, _, xu_cov = ugp_global_pol.get_posterior(massSlideWorld, x_mu_t, x_var_t)
#             # form joint state action distribution
#             xu_mu_t = np.append(x_mu_t, u_mu_t)
#             # xu_var_t = np.block([[x_var_t, np.zeros((dX,dU))],
#             #                     [np.zeros((dU,dX)), u_var_t]])
#             # TODO: xu_cov may not be correct so disable below and enable above later
#             xu_var_t = np.block([[x_var_t, xu_cov],
#                                  [xu_cov.T, u_var_t]])
#             X_mu_pred.append(x_mu_t)
#             X_var_pred.append(x_var_t)
#             # UT method for one step dynamics prediction
#             x_mu_t, x_var_t = model.predict(np.array([xu_mu_t]), np.array([xu_var_t]))
#             x_var_t = x_var_t + np.eye(dX, dX) * 1e-6
#             # unpack because our method takes a batch as input
#             x_mu_t = x_mu_t[0]
#             x_var_t = x_var_t[0]
#
#         # print(len(X_mu_pred), len(X_var_pred))
#         curr_res['pred_x_mu'] = X_mu_pred
#         curr_res['pred_x_var'] = X_var_pred
#
#         X_mu_pred = np.array(X_mu_pred)
#         P_sig_pred = np.zeros(H)
#         V_sig_pred = np.zeros(H)
#         for t in range(h):
#             P_sig_pred[t] = np.sqrt(np.diag(X_var_pred[t])[0])
#             V_sig_pred[t] = np.sqrt(np.diag(X_var_pred[t])[1])
#         P_mu_pred = X_mu_pred[:, :dP].reshape(-1)
#         V_mu_pred = X_mu_pred[:, dP:].reshape(-1)
#         tm = np.array(range(h))
#         plt.figure()
#         plt.title('Long-term prediction with BNN')
#         plt.subplot(121)
#         plt.xlabel('Time (s)')
#         plt.ylabel('Position (m)')
#         plt.plot(tm, P_mu_pred, marker='s', label='Pos mean', color='g', linewidth='2')
#         plt.fill_between(tm, P_mu_pred - P_sig_pred * 1.96, P_mu_pred + P_sig_pred * 1.96, alpha=0.2, color='g')
#         # plt.plot(tm, Xg[:H,0], linewidth='2')
#         plt.plot(tm, Xs_t_train[0, :H, :dP], ls='--', color='g', alpha=0.2, label='Training data')
#         for i in range(1, n_train):
#             plt.plot(tm, Xs_t_train[i, :H, :dP], ls='--', color='g', alpha=0.2)
#
#         plt.legend()
#         plt.subplot(122)
#         plt.xlabel('Time (s)')
#         plt.ylabel('Velocity (m/s)')
#         plt.plot(tm, V_mu_pred, marker='s', label='Vel mean', color='b', linewidth='2')
#         plt.fill_between(tm, V_mu_pred - V_sig_pred * 1.96, V_mu_pred + V_sig_pred * 1.96, alpha=0.2, color='b')
#         # plt.plot(tm, Xg[:H, 1], linewidth='2')
#         plt.plot(tm, Xs_t_train[0, :H, dP:], ls='--', color='b', alpha=0.2, label='Training data')
#         for i in range(1, n_train):
#             plt.plot(tm, Xs_t_train[i, :H, dP:], ls='--', color='b', alpha=0.2)
#         plt.legend()
#         plt.show(block=False)
#
#         # loglikelihood score
#         XUs_t_test = exp_data['XUs_t_test']
#         assert (XUs_t_test.shape[0] == n_test)
#         X_test_log_ll = np.zeros((h, n_test))
#         for t in range(h):  # one data point less than in XU_test
#             for i in range(n_test):
#                 XU_test = XUs_t_test[i]
#                 x_t = XU_test[t, :dX]
#                 x_mu_t = X_mu_pred[t]
#                 x_var_t = X_var_pred[t]
#                 X_test_log_ll[t, i] = sp.stats.multivariate_normal.logpdf(x_t, x_mu_t, x_var_t)
#
#         curr_res['ll_score'] = X_test_log_ll
#
#         bnn_res['horizon_{0}'.format(h)].append(curr_res)
#
#         del model
# np.save('bnn_res', bnn_res)


# # for multimodal output of BNN
# # here we adopt a simple propagation rule: treat each gaussian output in the ensemble as a local model by itself and maintain the ensemble all through the trajectory
# bnn_multm_res = dict()
# n_nets = 5
#
# for h in horizons:
#     bnn_multm_res['horizon_{0}'.format(h)] = []
#     for r in range(n_repeat):
#         print('Training for the {0}-th repetition...'.format(r))
#         curr_res = dict()
#
#         # for BNN
#         tf.reset_default_graph()
#         model = DynamicsPredictor(
#             model=BNNDynamics(layers=[3, 5, 2], n_nets=n_nets, name='bnn2d'))  # input 3d , output 2d
#
#         model.train(XU_t_train, X_t1_train)
#
#
#         # prediction
#         # we need a predictor indexing the i-th nn in the ensemble
#         class multimodal_predict(object):
#             def __init__(self, bnn_model, index):
#                 self.bnn_model = bnn_model
#                 self.index = index
#
#             def predict(self, inputs, return_std=True):
#                 means, variances = self.bnn_model.predict(inputs, factored=True)
#                 return means[self.index], np.sqrt(
#                     variances[self.index])  # return std instead, because get_posterior needs this
#
#
#         # prepare predictor for each model in the ensemble
#         predictors = [multimodal_predict(model.model.bnn_model, i_model) for i_model in range(n_nets)]
#
#         x_mu_t = exp_data['X0_mu']  # mean of initial state distr
#         x_var_t = np.diag(exp_data['X0_var'])
#         x_var_t[1, 1] = 1e-6  # setting small variance for initial vel    # TODO: cholesky failing for zero v0 variance
#
#         x_mu_t = np.array([x_mu_t.copy() for i in range(n_nets)])
#         x_var_t = np.array([x_var_t.copy() for i in range(n_nets)])
#
#         X_mu_pred = []  # list for collecting state mean
#         X_var_pred = []  # list for collecting state var
#         # for our baseline models to predict
#         for t in range(h):
#             X_mu_pred.append(x_mu_t.copy())
#             X_var_pred.append(x_var_t.copy())
#             for i_model in range(n_nets):
#                 # UT method on stochastic policy, policy is deterministic controller plus exploration noise
#                 u_mu_t, u_var_t, _, _, xu_cov = ugp_global_pol.get_posterior(massSlideWorld, x_mu_t[i_model],
#                                                                                  x_var_t[i_model])
#                 # form joint state action distribution
#                 xu_mu_t = np.append(x_mu_t[i_model], u_mu_t)
#                 # xu_var_t = np.block([[x_var_t, np.zeros((dX,dU))],
#                 #                     [np.zeros((dU,dX)), u_var_t]])
#                 # TODO: xu_cov may not be correct so disable below and enable above later
#                 xu_var_t = np.block([[x_var_t[i_model], xu_cov],
#                                      [xu_cov.T, u_var_t]])
#
#                 # UT method for one step dynamics prediction
#                 # x_mu_t, x_var_t = model.predict(np.array([xu_mu_t]), np.array([xu_var_t]))
#
#                 x_mu_t_new, x_var_t_new, _, _, _ = ugp_global_dyn.get_posterior(predictors[i_model], xu_mu_t, xu_var_t)
#                 x_var_t_new = x_var_t_new + np.eye(dX, dX)*1e-6
#
#                 # unpack because our method takes a batch as input
#                 x_mu_t[i_model] = x_mu_t_new
#                 x_var_t[i_model] = x_var_t_new
#
#         # print(len(X_mu_pred), len(X_var_pred))
#         curr_res['pred_x_mu'] = X_mu_pred
#         curr_res['pred_x_var'] = X_var_pred
#
#         # X_mu_pred = np.array(X_mu_pred)
#         # P_sig_pred = np.zeros(H)
#         # V_sig_pred = np.zeros(H)
#         # for t in range(h):
#         #     P_sig_pred[t] = np.sqrt(np.diag(X_var_pred[t])[0])
#         #     V_sig_pred[t] = np.sqrt(np.diag(X_var_pred[t])[1])
#         # P_mu_pred = X_mu_pred[:, :dP].reshape(-1)
#         # V_mu_pred = X_mu_pred[:, dP:].reshape(-1)
#         # tm = np.array(range(h))
#         # plt.figure()
#         # plt.title('Long-term prediction with mBNN')
#         # plt.subplot(121)
#         # plt.xlabel('Time (s)')
#         # plt.ylabel('Position (m)')
#         # plt.plot(tm, P_mu_pred, marker='s', label='Pos mean', color='g', linewidth='2')
#         # plt.fill_between(tm, P_mu_pred - P_sig_pred * 1.96, P_mu_pred + P_sig_pred * 1.96, alpha=0.2, color='g')
#         # # plt.plot(tm, Xg[:H,0], linewidth='2')
#         # plt.plot(tm, Xs_t_train[0, :H, :dP], ls='--', color='g', alpha=0.2, label='Training data')
#         # for i in range(1, n_train):
#         #     plt.plot(tm, Xs_t_train[i, :H, :dP], ls='--', color='g', alpha=0.2)
#         #
#         # plt.legend()
#         # plt.subplot(122)
#         # plt.xlabel('Time (s)')
#         # plt.ylabel('Velocity (m/s)')
#         # plt.plot(tm, V_mu_pred, marker='s', label='Vel mean', color='b', linewidth='2')
#         # plt.fill_between(tm, V_mu_pred - V_sig_pred * 1.96, V_mu_pred + V_sig_pred * 1.96, alpha=0.2, color='b')
#         # # plt.plot(tm, Xg[:H, 1], linewidth='2')
#         # plt.plot(tm, Xs_t_train[0, :H, dP:], ls='--', color='b', alpha=0.2, label='Training data')
#         # for i in range(1, n_train):
#         #     plt.plot(tm, Xs_t_train[i, :H, dP:], ls='--', color='b', alpha=0.2)
#         # plt.legend()
#         # plt.show(block=False)
#
#         # loglikelihood score
#         XUs_t_test = exp_data['XUs_t_test']
#         assert (XUs_t_test.shape[0] == n_test)
#         X_test_log_ll = np.zeros((h, n_test))
#         for t in range(h):  # one data point less than in XU_test
#             for i in range(n_test):
#                 XU_test = XUs_t_test[i]
#                 x_t = XU_test[t, :dX]
#                 x_mu_t = X_mu_pred[t]
#                 x_var_t = X_var_pred[t]
#                 # taking the mean of log of mean pdf
#                 X_test_log_ll[t, i] = np.log(
#                     np.mean([sp.stats.multivariate_normal.pdf(x_t, x_mu_t[j], x_var_t[j]) for j in range(n_nets)]))
#
#         curr_res['ll_score'] = X_test_log_ll
#
#         bnn_multm_res['horizon_{0}'.format(h)].append(curr_res)
#
#         del model
#
# np.save('bnn_multm_res', bnn_multm_res)


mgp_res = dict()
for h in horizons:
    mgp_res['horizon_{0}'.format(h)] = []
    for r in range(n_repeat):
        print('Training for the {0}-th repetition...'.format(r))
        curr_res = dict()

        # for mGP
        # tf.reset_default_graph()
        model = DynamicsPredictor(model=GaussianProcessDynamics(kern=NNFeaturedSE(3, [3, 3, 2])))
        model.train(XU_t_train, X_t1_train)

        # prediction
        x_mu_t = exp_data['X0_mu']  # mean of initial state distr
        x_var_t = np.diag(exp_data['X0_var'])
        x_var_t[1, 1] = 1e-6  # setting small variance for initial vel    # TODO: cholesky failing for zero v0 variance

        X_mu_pred = []  # list for collecting state mean
        X_var_pred = []  # list for collecting state var
        # for our baseline models to predict
        for t in range(h):
            # UT method on stochastic policy, policy is deterministic controller plus exploration noise
            u_mu_t, u_var_t, _, _, xu_cov = ugp_global_pol.get_posterior(massSlideWorld, x_mu_t, x_var_t)
            # form joint state action distribution
            xu_mu_t = np.append(x_mu_t, u_mu_t)
            # xu_var_t = np.block([[x_var_t, np.zeros((dX,dU))],
            #                     [np.zeros((dU,dX)), u_var_t]])
            # TODO: xu_cov may not be correct so disable below and enable above later
            xu_var_t = np.block([[x_var_t, xu_cov],
                                 [xu_cov.T, u_var_t]])
            X_mu_pred.append(x_mu_t)
            X_var_pred.append(x_var_t)
            # UT method for one step dynamics prediction
            x_mu_t, x_var_t = model.predict(np.array([xu_mu_t]), np.array([xu_var_t]))
            # unpack because our method takes a batch as input
            x_mu_t = x_mu_t[0]
            x_var_t = x_var_t[0]

        # print(len(X_mu_pred), len(X_var_pred))
        curr_res['pred_x_mu'] = X_mu_pred
        curr_res['pred_x_var'] = X_var_pred

        X_mu_pred = np.array(X_mu_pred)
        P_sig_pred = np.zeros(H)
        V_sig_pred = np.zeros(H)
        for t in range(h):
            P_sig_pred[t] = np.sqrt(np.diag(X_var_pred[t])[0])
            V_sig_pred[t] = np.sqrt(np.diag(X_var_pred[t])[1])
        P_mu_pred = X_mu_pred[:, :dP].reshape(-1)
        V_mu_pred = X_mu_pred[:, dP:].reshape(-1)
        tm = np.array(range(h))
        plt.figure()
        plt.title('Long-term prediction with mBNN')
        plt.subplot(121)
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.plot(tm, P_mu_pred, marker='s', label='Pos mean', color='g', linewidth='2')
        plt.fill_between(tm, P_mu_pred - P_sig_pred * 1.96, P_mu_pred + P_sig_pred * 1.96, alpha=0.2, color='g')
        # plt.plot(tm, Xg[:H,0], linewidth='2')
        plt.plot(tm, Xs_t_train[0, :H, :dP], ls='--', color='g', alpha=0.2, label='Training data')
        for i in range(1, n_train):
            plt.plot(tm, Xs_t_train[i, :H, :dP], ls='--', color='g', alpha=0.2)

        plt.legend()
        plt.subplot(122)
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.plot(tm, V_mu_pred, marker='s', label='Vel mean', color='b', linewidth='2')
        plt.fill_between(tm, V_mu_pred - V_sig_pred * 1.96, V_mu_pred + V_sig_pred * 1.96, alpha=0.2, color='b')
        # plt.plot(tm, Xg[:H, 1], linewidth='2')
        plt.plot(tm, Xs_t_train[0, :H, dP:], ls='--', color='b', alpha=0.2, label='Training data')
        for i in range(1, n_train):
            plt.plot(tm, Xs_t_train[i, :H, dP:], ls='--', color='b', alpha=0.2)
        plt.legend()
        plt.show(block=False)

        # loglikelihood score
        XUs_t_test = exp_data['XUs_t_test']
        assert (XUs_t_test.shape[0] == n_test)
        X_test_log_ll = np.zeros((h, n_test))
        for t in range(h):  # one data point less than in XU_test
            for i in range(n_test):
                XU_test = XUs_t_test[i]
                x_t = XU_test[t, :dX]
                x_mu_t = X_mu_pred[t]
                x_var_t = X_var_pred[t]
                X_test_log_ll[t, i] = sp.stats.multivariate_normal.logpdf(x_t, x_mu_t, x_var_t)

        curr_res['ll_score'] = X_test_log_ll

        mgp_res['horizon_{0}'.format(h)].append(curr_res)

        del model

np.save('mgp_res', mgp_res)

#visualize results
# bnn_res = np.load('bnn_res.npy').item()['horizon_74']
# bnn_multm_res = np.load('bnn_multm_res.npy').item()['horizon_39']
mgp_res = np.load('mgp_res.npy').item()['horizon_74']

#lets see scores
eval_horizons = [5, 9, 19, 29, 39]

# approaches = ['mGP', 'BNN', 'BNN-MM']
# approaches = ['BNN', 'BNN-MM']
approaches = ['BNN']

mgp_score_avg = []
mgp_score_std = []
bnn_score_avg = []
bnn_score_std = []
bnn_mm_score_avg = []
bnn_mm_score_std = []

for h in eval_horizons:
    mgp_score = []
    bnn_score = []
    for trial in range(1):
        #only look at the one repetition for the consistency with Shahbaz's criterion
        # bnn_score.append(np.mean(bnn_res[trial]['ll_score'][:h]))
        mgp_score.append(np.mean(mgp_res[trial]['ll_score'][:h]))
        # bnn_score = bnn_res[trial]['ll_score'][:h]
        # bnn_mm_score = bnn_multm_res[trial]['ll_score'][:h]
        mgp_score = mgp_res[trial]['ll_score'][:h]
    mgp_score_avg.append(np.mean(mgp_score))
    mgp_score_std.append(np.std(mgp_score))
    # bnn_score_avg.append(np.mean(bnn_score))
    # bnn_score_std.append(np.std(bnn_score))
    # bnn_mm_score_avg.append(np.mean(bnn_mm_score))
    # bnn_mm_score_std.append(np.std(bnn_mm_score))

print(mgp_score_avg, mgp_score_std)
# print(bnn_score_avg, bnn_score_std)
# print(bnn_mm_score_avg, bnn_mm_score_std)

#%matplotlib auto
fig = plt.figure()
ax = fig.add_subplot(111)

mgp_plt = ax.bar(eval_horizons, mgp_score_avg, yerr=mgp_score_std)
# bnn_plt = ax.bar(np.array(eval_horizons)-0.75, bnn_score_avg, yerr=bnn_score_std)
# bnn_mm_plt = ax.bar(np.array(eval_horizons)-1.5, bnn_mm_score_avg, yerr=bnn_mm_score_std)

ax.set_xlabel('Prediction Horizon', fontsize=16)
ax.set_ylabel('Log-likelihood Score', fontsize=16)

# ax.legend((mgp_plt, bnn_plt, bnn_mm_plt), approaches, fontsize=16)
# ax.legend((bnn_plt, bnn_mm_plt), approaches, fontsize=16)
# ax.legend((bnn_plt), approaches, fontsize=16)
ax.legend((mgp_plt), approaches, fontsize=16)
ax.grid()
plt.show()