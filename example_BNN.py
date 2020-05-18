import numpy as np
import pickle
import os
import dyn_pred

os.environ['CUDA_VISIBLE_DEVICES'] = ''
from dyn_pred.dynamics_predictor import DynamicsPredictor
from dyn_pred.bnn_dyn.bnn_dyn import BNNDynamics
from model_leraning_utils import yumi_joint_limits
import tensorflow as tf

logfile = "path to model learning data"
exp_data = pickle.load( open(logfile, "rb" ), encoding='latin1' )

exp_params = exp_data['exp_params']
dP = exp_params['dP'] # pos dim
dV = exp_params['dV'] # vel dim
dU = exp_params['dU'] # act dim
dX = dP+dV # state dim
T = exp_params['T']  # total time steps
dt = exp_params['dt'] # sampling time

XUs_t_train = exp_data['XUs_t_train'] # shape: n_train, T, dXU, state-action, sequential data
Xs_t1_train = exp_data['Xs_t1_train']
Xs_t_train = exp_data['Xs_t_train']
XU_t_train = XUs_t_train.reshape(-1, XUs_t_train.shape[-1]) # shape: n_train*T, dXU, state-action, sequential data
X_t1_train = Xs_t1_train.reshape(-1, Xs_t1_train.shape[-1]) # shape: n_train*T, dX, next state, sequential data
X_t_train = Xs_t_train.reshape(-1, Xs_t_train.shape[-1])
dX_t_train = X_t1_train - X_t_train

# use the joint limits to set prediction range. Without this long-term prediction can be unstable
yumi_joint_limits = np.array(yumi_joint_limits)
yumi_joint_max = yumi_joint_limits[:,1]
yumi_joint_min = yumi_joint_limits[:,0]
max_state_range = (yumi_joint_min, yumi_joint_max)

H = T  # prediction horizon, set to 1 for one-step prediction
num_traj_samples = (dP+dV+dU)*3     # number of Mote Carlo samples for uncertainty propagation
network_layer = [dX+dU, 64, 64, 64, dX]     # network structure for BNN
num_boot_straps = 10     # number of bootstraps used for BNN, higher the better but as many ANNs to train. 5 is good.

tf.reset_default_graph()
model = DynamicsPredictor(
    model=BNNDynamics(layers=network_layer, n_nets=num_boot_straps, name='bnn2d'))
model.train(XU_t_train, dX_t_train) # train the delta model

x_mu_t = exp_data['X0_mu']  # mean of initial state distr
x_var_t = np.diag(exp_data['X0_var'])   # cov of initial state distr
# pol is the policy and it should have the predict method. Can use the pol in model_learning_core
# generate num_traj_samples trajectory samples using the learned BNN for horizon H. H=1 if to be used in
# model_learning_core
traj_samples = model.sample_traj(x_mu_t, x_var_t, H=H, pol=pol.predict, U=None, n_particles=num_traj_samples,
                                 delta_model=True, unfactored=True, max_range=max_state_range)
# get trajectory distribution at every time step
traj_mean = np.mean(traj_samples, axis=0)
traj_std = np.sqrt(np.var(traj_samples, axis=0))    # only diagonal std
traj_covar = np.zeros((H, dX, dX))  # full covariance
for t in range(H):
    traj_covar[t] = np.cov(traj_samples[:, t, :], rowvar=False)

