import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from multidim_gp import MultidimGP
from model_leraning_utils import UGP
import time
import pickle
from blocks_sim import MassSlideWorld
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
from dyn_pred.dynamics_predictor import DynamicsPredictor
from dyn_pred.bnn_dyn.bnn_dyn import BNNDynamics
import tensorflow as tf
from model_leraning_utils import traj_with_globalgp

# np.random.seed(3)

#exp settings
n_repeat = 1  #number of training instances for nn approaches
horizons = [74]

#data and control things
# logfile = "/home/shahbaz/Research/Software/model_learning/Results/blocks_exp_preprocessed_data_rs_1_mm.p" # small data
# logfile = "/home/shahbaz/Research/Software/model_learning/Results/blocks_exp_preprocessed_data_rs_1_mm_bigdata.p"
logfile = "/home/shahbaz/Research/Software/model_learning/Results/blocks_exp_preprocessed_data_rs_1_mm_smalldata.p"
# result_file = "/home/shahbaz/Research/Software/model_learning/Results/results_blocks_bnn_bigdata.p"
result_file = "/home/shahbaz/Research/Software/model_learning/Results/results_blocks_bnn_smalldata.p"

exp_data = pickle.load( open(logfile, "rb" ), encoding='latin1' )

# bnn_results = {}
bnn_results = pickle.load( open(result_file, "rb" ), encoding='latin1' )
bnn_results['rmse'] = []
bnn_results['nll'] = []

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
print(XUs_t_train.shape)
XU_t_train = XUs_t_train.reshape(-1, XUs_t_train.shape[-1]) # shape: n_train*T, dXU, state-action, sequential data
Xs_t1_train = exp_data['Xs_t1_train']
X_t1_train = Xs_t1_train.reshape(-1, Xs_t1_train.shape[-1]) # shape: n_train*T, dX, next state, sequential data
Xs_t_train = exp_data['Xs_t_train']
Xs_t_test = exp_data['Xs_t_test']
X_t_test = Xs_t_test.reshape(-1, Xs_t_test.shape[-1])
XUs_t_test = exp_data['XUs_t_test']
Xs_t1_test = exp_data['Xs_t1_test']
X_t1_test = Xs_t1_test.reshape(-1, Xs_t1_test.shape[-1])
X_t_train = Xs_t_train.reshape(-1, Xs_t_train.shape[-1])
XU_t_test = XUs_t_test.reshape(-1, XUs_t_test.shape[-1])
U0_t_train = XUs_t_train[0, :, dX:dX+dU]
dX_t_train = X_t1_train - X_t_train
dX_t_test = X_t1_test - X_t_test

ugp_params = {
    'alpha': 1.,
    'kappa': 2.,
    'beta': 0.,
}

print('dX:{0}, dU:{1}'.format(dX, dU))

policy_params = exp_params['policy'] # TODO: the block_sim code assumes only 'm1' mode for control
expl_noise = policy_params['m1']['noise_pol']
H = T  # prediction horizon
delta_model = True

update_bnn_data = False
update_bnn_score = True

num_traj_samples = 50
#and world
# global gp long-term prediction
massSlideParams = exp_params['massSlide']
massSlideWorld = MassSlideWorld(**massSlideParams)
massSlideWorld.set_policy(policy_params)
massSlideWorld.reset()
mode = 'm1'  # only one mode for control no matter what X
pol = massSlideWorld.predict
ugp_global_dyn = UGP(dX + dU, **ugp_params) # initialize unscented transform for dynamics
ugp_global_pol = UGP(dX, **ugp_params) # initialize unscented transform for policy


x_mu_t = exp_data['X0_mu']  # mean of initial state distr
x_var_t = np.diag(exp_data['X0_var'])
x_var_t[1,1] = 1e-6   # setting small variance for initial vel    # TODO: cholesky failing for zero v0 variance
X0 = np.random.multivariate_normal(x_mu_t, x_var_t, 6)

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
    'degrees_of_freedom_prior': 2 + 2,
    'covariance_prior': None,
    'warm_start': False,
    'init_params': 'random',
}

layer_grid = [  [3, 8, 2],
                [3, 16, 2],
                [3, 32, 2],
                [3, 64, 2],
                [3, 128, 2],
                [3, 256, 2],
                [3, 8, 8, 2],
                [3, 16, 16, 2],
                [3, 32, 32, 2],
                [3, 64, 64, 2],
                [3, 128, 128, 2],
                [3, 8, 8, 8, 2],
                [3, 16, 16, 16, 2],
                [3, 32, 32, 32, 2],
                [3, 64, 64, 64, 2], ]
# n_repeat = len(layer_grid)
n_repeat = 10
network_layers = [3, 16, 16, 16, 2]
bnn_res = dict()
for h in horizons:
    bnn_res['horizon_{0}'.format(h)] = []
    for r in range(n_repeat):
        print('Training for the {0}-th repetition...'.format(r))
        curr_res = dict()

        # for BNN
        tf.reset_default_graph()
        # model = DynamicsPredictor(model=BNNDynamics(layers=[3, 32, 32, 2], n_nets=5, name='bnn2d'))  # input 3d , output 2d
        model = DynamicsPredictor(
            model=BNNDynamics(layers=network_layers, n_nets=5, name='bnn2d'))  # input 3d , output 2d

        start_time = time.time()
        if delta_model:
            model.train(XU_t_train, dX_t_train)
        else:
            model.train(XU_t_train, X_t1_train)
        train_time = time.time() - start_time
        bnn_results['train_time'] = train_time

        dX_t_test_pred, _ = model.model.bnn_model.predict(XU_t_test, factored=False)

        bnn_test_score = np.linalg.norm(dX_t_test - dX_t_test_pred)

        # print('bnn 1step test score for', layer_grid[r], bnn_test_score)

        start_time = time.time()
        # traj_samples = model.sample_traj(x_mu_t, x_var_t, H=T, pol=None, U=U0_t_train, n_particles=100)
        traj_samples = model.sample_traj(x_mu_t, x_var_t, H=h, pol=pol, U=None, n_particles=num_traj_samples, delta_model=delta_model, unfactored=False)
        traj_with_BNN.sample_trajs = traj_samples
        nll_mean, nll_std, rmse, X_test_log_ll = traj_with_BNN.estimate_gmm_traj_density(dpgmm_params, Xs_t_test, plot=False)
        pred_time = time.time() - start_time
        bnn_results['pred_time'] = pred_time
        print('NLL mean (mm): ', nll_mean, 'NLL std (mm): ', nll_std, 'RMSE:', rmse)
        # # plot taraj samples
        # plt.figure()
        # plt.subplot(121)
        # plt.title('Pos')
        # plt.plot(range(T), traj_samples[:, :, 0].T, color='g', alpha=0.1)
        # for i in range(0, n_train):
        #     plt.plot(range(T), Xs_t_train[i, :T, :dP], ls='--', color='g', alpha=0.2)
        # plt.subplot(122)
        # plt.title('Vel')
        # plt.plot(range(T), traj_samples[:, :, 1].T, color='b', alpha=0.1)
        # for i in range(0, n_train):
        #     plt.plot(range(T), Xs_t_train[i, :T, dP:], ls='--', color='b', alpha=0.2)
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
            bnn_results['density_est'] = traj_with_BNN
            pickle.dump(bnn_results, open(result_file, "wb"), protocol=2)

# np.save('bnn_res', bnn_res)
#
# #visualize results
# bnn_res = np.load('bnn_res.npy').item()['horizon_74']
#
# #lets see scores
# eval_horizons = [5, 34, 74]
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