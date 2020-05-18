'''
A wrapper of BNN for a consistent interface with variance propagation
'''

from __future__ import print_function
import tensorflow as tf

import numpy as np

# from . DotmapUtils import *
from bnn_dyn.FC import FC
from bnn_dyn.BNN import BNN
from model_leraning_utils import yumi_joint_limits

from dotmap import DotMap

class BNNDynamics(object):
    def __init__(self, layers, n_nets=5, name="bnn_model", sess=None, output_activation=None):
        self.n_ensemble = n_nets

        if isinstance(layers, str):
            model_dir = layers
            load_model = True
        else:
            load_model = False
            model_dir = None

        params = DotMap(
            name=name, num_networks=n_nets,
            sess=sess, load_model=load_model,
            model_dir=model_dir
        )
        self.bnn_model = BNN(params)

        #construct layers if not loaded
        if not load_model:
            #the layers indicate dimensions of each layers
            self.input_dim = layers[0]
            for i in range(len(layers)-2):
                if i == 0:
                    self.bnn_model.add(FC(layers[1], input_dim=layers[0], activation="swish", weight_decay=0.000025))
                else:
                    self.bnn_model.add(FC(layers[i+1], activation="swish", weight_decay=0.000075))
            #output, linear output
            self.bnn_model.add(FC(layers[-1], activation=output_activation, weight_decay=0.0001))
            self.bnn_model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})
        else:
            self.input_dim = self.bnn_model.layers[0].input_dim

        #put training parameters here for further tunning
        self.batch_size=32
        self.epochs=100
        self.hide_progress=False
        self.holdout_ratio=0.0
        self.max_logging=5000
        return
    
    def train(self, inputs, outputs):
        self.bnn_model.train(inputs, outputs, self.batch_size, self.epochs, 
            self.hide_progress, self.holdout_ratio, self.max_logging)

        return
    
    def predict_f(self, inputs):
        return self.bnn_model.predict(inputs, factored=False)

    def predict_trajs(self, X0, U=[None]*9, n_particles=20, prop='TSInf', state_proc=lambda x:x, state_postproc=lambda x, pred:pred, prob=True):
        '''
        predict trajectories for given init states
        following the trajectory sampling schemes like TS1 and TSInf but without backpropagate differentiability
        X0:             (batch_size, x_dim)
        U:              (T, u_dim), control can be augmented with predicted state x, could be None if dynamics is passive. Note the same control applies to entire state batch
        n_particles:    number of particles to obtain the empirical distribution of trajectories
        prop:           type of particle propagation, TS1/TSInf
        state_proc:     process state before it feeds to the dynamics model
        state_postproc: customized process to ensure the next iteration is compatible, e.g., if we pred velocity based on curr pos+vel, we need to augment a pred pos to obtain another pos+vel 

        return:
        trajs:          (batch_size, n_particles, T, x_dim)
        '''
        T = len(U)
        bs = X0.shape[0]
        #preprocess x0
        X_prev = np.vstack([np.tile(x0, (n_particles, 1)) for x0 in X0])    #(batch_size*n_particles, x_dim)
        trajs = np.empty((bs, n_particles, T+1, X_prev.shape[-1]))
        
        for t in range(T):
            trajs[:, :, t, :] = np.reshape(X_prev, (-1, n_particles, X_prev.shape[-1]))
            X_proc = state_proc(X_prev)

            if prop == 'TS1':
                #need to resample   
                X_proc = np.reshape(X_proc, (-1, n_particles, X_proc.shape[-1]))    #(batch_size, n_particles, x_dim)
                #for each batch, generate a permutation
                sort_idx = np.array([np.random.permutation(n_particles) for _ in range(X_proc.shape[0])])      #(batch_size, n_particles)
                #tmp = np.tile(np.arange(X_proc.shape[0])[:, None], [1, n_particles])[:, :, None]        #(batch_size) --> (batch_size, 1) --> (batch_size, n_particles) --> (bs, np, 1)
                #have some issues here for TS1, what to use in numpy for tf.gather_nd?
                #idxs = np.concatenate([tmp, sort_idx[:, :, None]], axis=-1)                         #(bs, np, 2)   
                tmp = np.tile(np.arange(X_proc.shape[0])[:, None], [1, n_particles]).flatten()        #(batch_size) --> (batch_size, 1) --> (batch_size, n_particles) --> (batch_size*n_particles)
                
                X_proc = X_proc[tmp, sort_idx.flatten(), :]                                            #(bs*np, xdim)
                #X_proc = np.reshape(X_proc, (-1, X_proc.shape[-1]))                                 #(bs*np, xdim)

            X_proc = self._expand_to_ts_format(X_proc, n_particles)
            #see if there is control to apply
            if U[t] is not None:
                #combine this control
                U_proc = self._expand_to_ts_format(np.tile(U[t], (bs*n_particles, 1)), n_particles)
                inputs = np.concatenate((X_proc, U_proc), axis=-1)
            else:
                inputs = X_proc

            mean, var = self.bnn_model.predict(inputs, factored=True)
            if prob:
                #take a sample if we expect a probabilistic output
                pred = mean + np.random.randn(*(mean.shape)) * np.sqrt(var)
            else:
                #take mean as the decision making here
                pred = mean

            pred = self._flatten_to_matrix(pred, n_particles)
            if prop == 'TS1':
                '''
                reverse that process
                '''
                pred = np.reshape(pred, (-1, n_particles, pred.shape[-1]))      #(bs, np, pdim)
                sort_idx_reversed = np.empty(sort_idx.shape, dtype=int)
                for b in range(sort_idx.shape[0]):
                    for i in range(sort_idx.shape[1]):
                        sort_idx_reversed[b][sort_idx[b][i]] = i
                #sort_idx = np.array([[b_idx for i, _ in enumerate(b_idx)] for b_idx in sort_idx])
                #idxs = np.concatenate([tmp, sort_idx[:, :, None]], axis=-1)
                #pred = np.take(pred, idxs)                                                      
                #pred = np.reshape(pred, (-1, pred.shape[-1]))
                pred = pred[tmp, sort_idx_reversed.flatten(), :]

            X_prev = state_postproc(X_prev, pred)
        
        trajs[:, :, T, :] = np.reshape(X_prev, (-1, n_particles, X_prev.shape[-1]))
        return trajs

    def predict_trajs_pol(self, mu_x0, var_x0, H=10, pol=None, U=None, n_particles=20, prop='TSInf', delta_model=False, unfactored=False, max_range=None):
        T = H
        X_prev = np.random.multivariate_normal(mu_x0, var_x0, n_particles)
        trajs = np.empty((n_particles, T, X_prev.shape[-1]))

        for t in range(T):
            trajs[:, t, :] = np.reshape(X_prev, (n_particles, X_prev.shape[-1]))
            # X_proc = state_proc(X_prev)
            X_proc = X_prev

            if not unfactored:
                X_proc = self._expand_to_ts_format(X_proc, n_particles)
            # see if there is control to apply
            if U is not None:
                # combine this control
                if not unfactored:
                    U_proc = self._expand_to_ts_format(np.tile(U[t], (n_particles, 1)), n_particles)
                else:
                    U_proc = np.tile(U[t], (n_particles, 1))
                inputs = np.concatenate((X_proc, U_proc), axis=-1)
            elif pol is not None:
                if not unfactored:
                    U_mu, U_std = pol(self._flatten_to_matrix(X_proc, n_particles), t=t)
                else:
                    U_mu, U_std = pol(X_proc, t=t)
                U_proc = np.random.normal(U_mu, U_std)
                if not unfactored:
                    U_proc = self._expand_to_ts_format(U_proc, n_particles)
                inputs = np.concatenate((X_proc, U_proc), axis=-1)
            else:
                inputs = X_proc

            if not unfactored:
                mean, var = self.bnn_model.predict(inputs, factored=True)
            else:
                mean, var = self.bnn_model.predict(inputs, factored=False)
            pred = mean + np.random.randn(*(mean.shape)) * np.sqrt(var)
            if not unfactored:
                pred = self._flatten_to_matrix(pred, n_particles)

            # X_prev = state_postproc(X_prev, pred)
            if delta_model:
                # if max_range is not None:  # sets the predictions outside the expected value to zero.
                #     pred[pred > max_range[1]] = 0.
                #     pred[pred < max_range[0]] = 0.
                X_prev = X_prev + pred
            else:
                X_prev = pred
            if max_range is not None:  # restrict predicted values to within the joint range
                X_prev = np.clip(X_prev, max_range[0], max_range[1])

        return trajs
    
    def _expand_to_ts_format(self, mat, n_particles):
        xdim = mat.shape[-1]
        return np.reshape(np.transpose(np.reshape(mat, (-1, self.n_ensemble, n_particles//self.n_ensemble, xdim)), [1, 0, 2, 3]), [self.n_ensemble, -1, xdim])
    
    def _flatten_to_matrix(self, ts_fmt_arr, n_particles):
        xdim = ts_fmt_arr.shape[-1]
        return np.reshape(np.transpose(np.reshape(ts_fmt_arr, [self.n_ensemble, -1, n_particles//self.n_ensemble, xdim]), [1, 0, 2, 3]), [-1, xdim])

    
