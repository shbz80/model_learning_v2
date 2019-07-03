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
from dyn_pred.gp_dyn.gp_dyn import GaussianProcessDynamics
from dyn_pred.gp_dyn.manifold_gp.gpflow_nn_ker import NNFeaturedSE
import gpflow
import tensorflow as tf
from model_leraning_utils import traj_with_globalgp

# np.random.seed(3)

#exp settings
n_repeat = 1  #number of training instances for nn approaches
horizons = [74]

#data and control things
# logfile = "/home/shahbaz/Research/Software/model_learning/Results/blocks_exp_preprocessed_data_rs_1_mm.p"   # small data
# logfile = "/home/shahbaz/Research/Software/model_learning/Results/blocks_exp_preprocessed_data_rs_1_mm_bigdata.p"
# result_file = "/home/shahbaz/Research/Software/model_learning/Results/results_blocks_mgp_bigdata.p"
logfile = "/home/shahbaz/Research/Software/model_learning/Results/blocks_exp_preprocessed_data_rs_1_mm_smalldata.p"
result_file = "/home/shahbaz/Research/Software/model_learning/Results/results_blocks_mgp_smalldata.p"

exp_data = pickle.load( open(logfile, "rb" ), encoding='latin1' )

# mgp_results = {}
mgp_results = pickle.load( open(result_file, "rb" ), encoding='latin1' )
mgp_results['rmse'] = []
mgp_results['nll'] = []

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
update_mgp_data = False
update_mgp_score = True

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

traj_with_mGP = traj_with_globalgp(x_mu_t, x_var_t, None, None, dlt_mdl=delta_model)


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

layer_grid = [  [3, 8, 8, 2],
                [3, 16, 16, 2],
                [3, 32, 32, 2],
                # [3, 64, 64, 2],
                [3, 8, 8, 8, 2],
                [3, 16, 16, 16, 2],
                [3, 32, 32, 32, 2],
                                     ]
# n_repeat = len(layer_grid)
n_repeat = 10
network_layers = [3, 32, 32, 2]

massSlideWorld.reset()
mgp_res = dict()
for h in horizons:
    mgp_res['horizon_{0}'.format(h)] = []
    for r in range(n_repeat):
        print('Training for the {0}-th repetition...'.format(r))
        curr_res = dict()

        # for mGP
        # tf.reset_default_graph()
        model = DynamicsPredictor(model=GaussianProcessDynamics(kern=NNFeaturedSE(3, network_layers)))

        start_time = time.time()
        if delta_model:
            model.train(XU_t_train, dX_t_train)
        else:
            model.train(XU_t_train, X_t1_train)
        training_time = time.time() - start_time
        mgp_results['train_time'] = time.time() - start_time
        print('mGP training time', training_time)

        dX_t_test_pred, _ = model.model.predict_f(XU_t_test)

        mgp_test_score = np.linalg.norm(dX_t_test - dX_t_test_pred)

        print('mgp 1step test score:', mgp_test_score)


        # prediction
        x_mu_t = exp_data['X0_mu']  # mean of initial state distr
        x_var_t = np.diag(exp_data['X0_var'])
        x_var_t[1, 1] = 1e-6  # setting small variance for initial vel    # TODO: cholesky failing for zero v0 variance

        traj_samples = np.zeros((num_traj_samples, h, dX))
        start_time = time.time()
        for s in range(num_traj_samples):
            traj_samples[s][0] = np.random.multivariate_normal(x_mu_t, x_var_t)
            for t in range(1, h):
                x = traj_samples[s][t - 1]
                x = x.reshape(-1)
                um, uv = pol(x.reshape(1, -1))
                um = np.asscalar(um)
                ustd = np.asscalar(uv)
                u = np.random.normal(um, ustd)
                xu = np.append(x, u)
                mu, var = model.model.predict_f(xu.reshape(1, -1))
                mu = mu.reshape(-1)
                var = var.reshape(-1)
                var = np.diag(var)
                if not delta_model:
                    traj_samples[s][t] = np.random.multivariate_normal(mu, var)
                else:
                    traj_samples[s][t] = np.random.multivariate_normal(mu, var) + traj_samples[s][t - 1]
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

        traj_with_mGP.sample_trajs = traj_samples
        nll_mean, nll_std, rmse, X_test_log_ll = traj_with_mGP.estimate_gmm_traj_density(dpgmm_params, Xs_t_test, plot=False)
        mgp_results['pred_time'] = time.time() - start_time
        print('NLL mean (mm): ', nll_mean, 'NLL std (mm): ', nll_std, 'RMSE:', rmse)


        curr_res['ll_score'] = X_test_log_ll

        mgp_res['horizon_{0}'.format(h)].append(curr_res)

        del model

        if update_mgp_score:
            mgp_results['rmse'].append(rmse)
            mgp_results['nll'].append((nll_mean, nll_std))
            pickle.dump(mgp_results, open(result_file, "wb"), protocol=2)
        if update_mgp_data:
            mgp_results['traj_samples'] = traj_samples
            mgp_results['density_est'] = traj_with_mGP
            pickle.dump(mgp_results, open(result_file, "wb"), protocol=2)
#
# np.save('mgp_res', mgp_res)
#
# #visualize results
# mgp_res = np.load('mgp_res.npy').item()['horizon_74']
#
# #lets see scores
# eval_horizons = [5, 34, 74]
# approaches = ['mGP']
#
# mgp_score_avg = []
# mgp_score_std = []
#
# for h in eval_horizons:
#     mgp_score = []
#     for trial in range(1):
#         mgp_score = mgp_res[trial]['ll_score'][:h]
#     mgp_score_avg.append(np.mean(mgp_score))
#     mgp_score_std.append(np.std(mgp_score))
#
# print(mgp_score_avg, mgp_score_std)
#
# #%matplotlib auto
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# mgp_plt = ax.bar(eval_horizons, mgp_score_avg, yerr=mgp_score_std)
# ax.set_xlabel('Prediction Horizon', fontsize=16)
# ax.set_ylabel('Log-likelihood Score', fontsize=16)
# ax.legend((mgp_plt), approaches, fontsize=16)
# ax.grid()
# plt.show()
None