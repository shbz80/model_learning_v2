import gpflow
import tensorflow as tf
import numpy as np

from gpflow import Parameterized, Param

from gpflow.kernels import RBF

def xavier(dim_in, dim_out):
    return np.random.randn(dim_in, dim_out)*(2./(dim_in+dim_out))**0.5

class NN(Parameterized):
    def __init__(self, dims):
        Parameterized.__init__(self)
        self.dims = dims
        for i, (dim_in, dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            setattr(self, 'W_{}'.format(i), Param(xavier(dim_in, dim_out)))
            setattr(self, 'b_{}'.format(i), Param(np.zeros(dim_out)))

    def forward(self, X):
        if X is not None:
            for i in range(len(self.dims) - 1):
                W = getattr(self, 'W_{}'.format(i))
                b = getattr(self, 'b_{}'.format(i))
                X = tf.nn.tanh(tf.matmul(X, W) + b)
            return X


class NN_RBF(RBF):
    def __init__(self, nn, *args, **kw):
        RBF.__init__(self, *args, **kw)
        self.nn = nn
    
    def square_dist(self, X, X2):
        return RBF.square_dist(self, self.nn.forward(X), self.nn.forward(X2))

from tensorflow.python.ops import variables

net = NN([5, 10, 2])  # for 5D inputs and a 2D GP
kern = NN_RBF(net, 2)

train_data_x = np.random.randn(100, 5)
train_data_y = np.random.randn(100, 1)


m = gpflow.models.GPR(train_data_x, train_data_y, kern=kern)
m.compile()
# print(m.read_trainables())
print(m.compute_log_likelihood())
opt = gpflow.train.ScipyOptimizer()
opt.minimize(m)
print(variables.trainable_variables())
# print(m.read_trainables())
print(m.compute_log_likelihood())