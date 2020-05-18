"""
model to learn and inference upon input and its uncertainty
"""
from __future__ import print_function
# import tensorflow as tf

import numpy as np
import scipy

class DynamicsPredictor(object):
    def __init__(self, model=None):
        #model must support train and predict functions, and deal with uncertainty
        self.model = model
        #parameters for unscented kf like propagation with variance
        #see unscented kf paper for more details
        self.alpha=1e-3
        self.kappa=0
        self.beta=2.
    

    def train(self, inputs, outputs):
        #for P(outputs|inputs)
        if self.model is not None:
            self.model.train(inputs, outputs)
        else:
            print('No model is initialized.')
        return
    
    def _unscented_get_sigma_points(self, mu, var):
        '''
        deterministic process to pick sigma points to propagate to a nonlinear transformation process
        '''
        # note this only works for one point so mu must be (input_dim,) and var must be (input_dim, input_dim)
        L=self.model.input_dim
        N=2*L+1 #number of sigma points to evaluate the nonlinear transformation
        assert(mu.shape == (L,))
        assert (var.shape == (L, L))
        alpha = self.alpha
        kappa = self.kappa
        beta = self.beta
        sigmaMat = np.zeros((N, L))

        Lambda = (alpha**2) * (L + kappa) - L
        sigmaMat[0, :] = mu
        try:
            chol = scipy.linalg.cholesky((L + Lambda)*var, lower=True)
        except np.linalg.LinAlgError:
            assert(False)
        for i in range(1, L+1):
            sigmaMat[i, :] = mu + chol[:,i-1]
            sigmaMat[L+i, :] = mu - chol[:, i-1]

        #weights for mean and var
        W_mu = np.zeros(N)
        Wi = 1. / (2. * (L + Lambda))
        W_mu.fill(Wi)
        W_mu[0] = Lambda / (L + Lambda)

        W_var = np.zeros(N)
        W_var.fill(Wi)
        W_var[0] = W_mu[0] + (1. - alpha**2 + beta)
        return sigmaMat, W_mu, W_var
    
    def _predict_with_var_propagated(self, input, input_var):
        #an internal function to propagate both input_mean and input_var
        #adapted from Shahbaz's repository
        def get_posterior(mu, var):
            '''
            Compute and return the output distribution along with the propagated sigma points
            :param fn: the nonlinear function through which to propagate
            :param mu: mean of the input
            :param var: variance of the output
            :return:
                    Y_mu_post: output mean
                    Y_var_post: output variance
                    Y_mu: transformed sigma points
                    Y_var: gp var for each transformed points
                    XY_cross_cov: cross covariance between input and output
            '''
            sigmaMat, W_mu, W_var = self._unscented_get_sigma_points(mu, var)
            Y_mu, Y_var = self.model.predict_f(sigmaMat) # same signature as the predict function of gpr but can be
                                                                # any nonlinear function
            N, Do = Y_mu.shape                              # number of sigma points and output dimension...
            #Y_var = Y_var.reshape(N,Do)
            Y_mu_post = np.average(Y_mu, axis=0, weights=W_mu)    # DX1
            # Y_mu_post = Y_mu[0]
            Y_var_post = np.zeros((Do,Do))
            for i in range(N):
                y = Y_mu[i] - Y_mu_post
                yy_ = np.outer(y, y)
                Y_var_post += W_var[i]*yy_
            #Y_var_post = np.diag(np.diag(Y_var_post))     # makes it worse
            Y_var_post += np.diag(Y_var[0])                 #add the predicted variance of mean, see UGP paper, is this a heuristic?
            # if Do == 1:
            #     Y_mu_post = np.asscalar(Y_mu_post)
            #     Y_var_post = np.asscalar(Y_var_post)
            # compute cross covariance between input and output
            Di = mu.shape[0]
            XY_cross_cov = np.zeros((Di, Do))
            for i in range(N):
                y = Y_mu[i] - Y_mu_post
                x = sigmaMat[i] - mu
                xy_ = np.outer(x, y)
                XY_cross_cov += W_var[i] * xy_
            return Y_mu_post, Y_var_post, Y_mu, Y_var, XY_cross_cov
        
        #didnt find a way to do two list comprehension in one iterate, use loop for now, :-(
        #or maybe we should repeat weights to exploit a multidim matrix operation in one shot...
        pred_mean = []
        pred_var = []
        
        for i_mu, i_var in zip(input, input_var):
            #ignore covariance between input and output now
            o_post_mu, o_post_var, _, _, _  = get_posterior(i_mu, i_var)
            pred_mean.append(o_post_mu)
            pred_var.append(o_post_var)
        
        return np.array(pred_mean), np.array(pred_var)
    
    def predict(self, X, X_Var=None, U=None, U_Var=None, XU_CrossVar=None):
        #core inference function
        #U: control input: assume a passive dynamics if control is not given (padding 0?)
        #U_Var: variance of control input: do we need uncertainty about control as well for probabilistic policies?
        #XU_CrossVar: cross variance between X and U inputs
        #simply a GPR regression if X_Var is None, in which case, returns the predicted mean and covariance
        #otherwise, use Unscented KF like techniques to propagate covariance as well
        # X.shape[0] == X_Var.shape[0]: number of input instances
        # X.shape[1]: 
        assert(self.model is not None)
        
        if U is not None:
            assert(U.shape[0]==X.shape[0])
            assert(U.shape[1]+X.shape[1]==self.model.input_dim)
        else:
            U=np.zeros((X.shape[0], self.model.input_dim-X.shape[1]))
        
        if X_Var is not None:
            assert(X_Var.shape[0]==X.shape[0])
            assert(X_Var.shape[1]==X_Var.shape[2] and X_Var.shape[1]==X.shape[1])
            
            if U_Var is not None:
                assert(U_Var.shape[0]==U.shape[0])
                if len(U_Var.shape)==2:
                    assert(U_Var.shape[1]==U.shape[1])
                    #diagnolize the flattened representation
                    U_Var = [np.diag(U_Var) for v_u in U_Var]
                else:
                    assert(U_Var.shape[1]==U_Var.shape[2] and U_Var.shape[1]==U.shape[1])
            else:
                #have a small isotropic variance by default
                #well am wondering if it would be possible do cholesky only X in this case but I guess might break UKF to
                #generate particles from a degenerated Gaussian, or should we assume a control-affine dynamics here?
                U_Var=np.array([np.eye(U.shape[1])*1e-6 for i in range(U.shape[0])]) # TODO: changed Hang's 1e-4 to 1e-6
            if XU_CrossVar is not None:
                assert(XU_CrossVar.shape[0]==U.shape[0])
                assert(XU_CrossVar.shape[1]==X.shape[1] and XU_CrossVar.shape[2]==U.shape[1])
            else:
                XU_CrossVar=np.array([np.zeros((X.shape[1], U.shape[1])) for i in range(U.shape[0])])
        
        #combine X and U to give input
        combined_input = np.hstack([X, U])
        if X_Var is not None:
            #combined_input_var = np.array([scipy.linalg.block_diag(v_x, v_u) for v_x, v_u in zip(X_Var, U_Var)])
            combined_input_var = np.array([np.block([[v_x, v_xu],
                                                      [v_xu.T, v_u]]) for v_x, v_u, v_xu in zip(X_Var, U_Var, XU_CrossVar)])
        else:
            combined_input_var = None

        if combined_input_var is None:
            #simply do a normal probabilistic regression
            pred_mean, pred_var = self.model.predict_f(combined_input)
            #note the pred_var is of a flattened shape because the conditional independence
            pred_var = np.array([np.diag(v) for v in pred_var])
        else:
            #use unscented kf to propagate combined_input_var as well...
            pred_mean, pred_var = self._predict_with_var_propagated(combined_input, combined_input_var)
        
        return pred_mean, pred_var
    
    def predict_n_steps(self, n_steps, X0, X0_Var=None, U=None, U_Var=None):
        #recursively applying the dynamics for a long-term prediction from X0. this could be really bad
        #because we take an iid assumption about input/output in training
        #it might make sense to support auto-regressive prediction as well, thinking about automatically moving the state window...
        #in that case, we need the cross variance between input and output
        #return: lists of predictive mean and variance
        #           pred_means[i], pred_vars[i] at the (i+1)-th step
        assert(self.model is not None)
        pred_means = []
        pred_vars = []
        next_input = X0
        next_input_var = X0_Var
        for i in range(n_steps):
            if U is None:
                pred_mu, pred_var = self.predict(next_input, next_input_var, None, None)
            else:
                if U_Var is None:
                    pred_mu, pred_var = self.predict(next_input, next_input_var, U[i], None)
                else:
                    pred_mu, pred_var = self.predict(next_input, next_input_var, U[i], U_Var[i])
            pred_means.append(pred_mu)
            pred_vars.append(pred_var)

            next_input = pred_means[-1]
            next_input_var = pred_vars[-1]
        
        return pred_means, pred_vars

    # def sample_traj(self, X0, U=[None]*9, n_particles=20, prop='TSInf', state_proc=lambda x:x, state_postproc=lambda x, pred:pred, prob=True):
    #     if self.model is not None:
    #         return self.model.predict_trajs_pol(X0, U, n_particles=n_particles, prop=prop, state_proc=state_proc, state_postproc=state_postproc, prob=prob)
    #     else:
    #         print('No model is initialized.')
    #         return None

    def sample_traj(self, mu_x0, var_x0, H=10, pol=None, U=None, n_particles=20, prop='TSInf', delta_model=False, unfactored=False, max_range=None):
        if self.model is not None:
            return self.model.predict_trajs_pol(mu_x0, var_x0, H=H, pol=pol, U=U, n_particles=n_particles, prop=prop, delta_model=delta_model, unfactored=unfactored, max_range=max_range)
        else:
            print('No model is initialized.')
            return None