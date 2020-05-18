'''
An implementation of GPflow kernel with NN features, referring to manifold GP
'''

import tensorflow as tf
import gpflow
import numpy as np

def xavier(dim_in, dim_out):
    return np.random.randn(dim_in, dim_out)*(2./(dim_in+dim_out))**0.5

#fully-connected NN feature for usage 
class FCNN(gpflow.Parameterized):
    def __init__(self, feature_dims, output_nonlinearity=None):
        gpflow.Parameterized.__init__(self)
        self.feature_dims = feature_dims
        self.output_nonlinearity = output_nonlinearity
        for i, (dim_in, dim_out) in enumerate(zip(feature_dims[:-1], feature_dims[1:])):
            setattr(self, 'FCNN_Weight_{}'.format(i), gpflow.Param(xavier(dim_in, dim_out)))
            setattr(self, 'FCNN_Bias_{}'.format(i), gpflow.Param(np.zeros(dim_out)))

    def forward(self, X):
        if X is not None:
            for i in range(len(self.feature_dims) - 1):
                W = getattr(self, 'FCNN_Weight_{}'.format(i))
                b = getattr(self, 'FCNN_Bias_{}'.format(i))
                X = tf.matmul(X, W) + b
                #use elu as default nonlinearity for intermediate hidden layers
                if i < len(self.feature_dims) - 2:
                    X = tf.nn.elu(X)
                else:
                    if self.output_nonlinearity is not None:
                        X = self.output_nonlinearity(X)
            return X

class NNFeaturedSE(gpflow.kernels.Kernel):
    def __init__(self, input_dim, feature_dims=None, output_nonlinearity=None, variance=1.0, lengthscales=1.0, ARD=True):
        super().__init__(input_dim=input_dim)
        self.variance = gpflow.Param(1.0, transform=gpflow.transforms.positive)
        self.input_dim = input_dim

        if ARD:
            if feature_dims is not None:
                self.lengthscales = gpflow.Param([1.0] * feature_dims[-1], transform=gpflow.transforms.positive,
                    dtype=gpflow.settings.float_type)
            else:
                self.lengthscales = gpflow.Param([1.0] * self.input_dim, transform=gpflow.transforms.positive,
                            dtype=gpflow.settings.float_type)
        else:
            self.lengthscales = gpflow.Param(1.0, transform=gpflow.transforms.positive,
                            dtype=gpflow.settings.float_type)

        if feature_dims is not None:
            assert feature_dims[0] == self.input_dim
            self.nn_transform = FCNN(feature_dims, output_nonlinearity)
        else:
            self.nn_transform = None
        
        return

    def _nn_featured_scaled_square_dist(self, X, X2):
        '''
        return M(X)
        '''
        M = X
        M2 = X2

        if self.nn_transform is not None:
            #build features for X1 and X2 (if not None), use elu nonlinearity
            M = self.nn_transform.forward(X)
            M2 = self.nn_transform.forward(X2)
        
        #now lets calculate scaled square distance, see the gpflow example
        M = M / self.lengthscales
        Ms = tf.reduce_sum(tf.square(M), axis=-1, keepdims=True)

        if M2 is None:
            dist = -2 * tf.matmul(M, M, transpose_b=True)
            dist += Ms + tf.matrix_transpose(Ms)
            return dist

        M2 = M2 / self.lengthscales
        M2s = tf.reduce_sum(tf.square(M2), axis=-1, keepdims=True)
        dist = -2 * tf.matmul(M, M2, transpose_b=True)
        dist += Ms + tf.matrix_transpose(M2s)
        return dist
    
    def nn_featured_scaled_square_dist(self, X, X2):
        return self._nn_featured_scaled_square_dist(X, X2)
    
    @staticmethod
    def _clipped_sqrt(r2):
        # Clipping around the (single) float precision which is ~1e-45.
        return tf.sqrt(tf.maximum(r2, 1e-40))
    
    
    def K(self, X, X2=None):
        return self.K_r2(self.nn_featured_scaled_square_dist(X, X2))

    def Kdiag(self, X, presliced=False):
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))
    
    def K_r2(self, r2):
        """
        Returns the kernel evaluated on `r2`, which is the scaled squared distance.
        Will call self.K_r(r=sqrt(r2)), or can be overwritten directly (and should operate element-wise on r2).
        """
        r = self._clipped_sqrt(r2)
        return self.variance * tf.exp(-r / 2.)