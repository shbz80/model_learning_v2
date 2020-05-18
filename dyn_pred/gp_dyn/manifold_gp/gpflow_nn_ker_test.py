'''
a test for gpflow_nn_ker
'''
import tensorflow as tf
import gpflow
import gpflow_nn_ker
import numpy as np

import matplotlib.pyplot as plt

def gpflow_nn_ker_test():
    '''
    the step function example
    '''
    custom_config = gpflow.settings.get_settings()
    custom_config.verbosity.optimisation_verb = True

    step_func = lambda x: np.where(x>0, 1.0, 0.0)
    noise_sigma = 0.01

    #for training data
    num_train = 100
    training_data_x = np.random.randn(num_train, 1)
    training_data_y = step_func(training_data_x) + noise_sigma * np.random.randn(num_train, 1)
    # print(training_data_x.shape, training_data_y.shape)
    
    # with gpflow.defer_build():
    m = gpflow.models.GPR(training_data_x, training_data_y, kern=gpflow_nn_ker.NNFeaturedSE(1, [1, 6, 2], 'sigmoid'))
    # m = gpflow.models.GPR(training_data_x, training_data_y, kern=gpflow.kernels.SquaredExponential(1))
    print(m.read_trainables())
    m.compile()
    opt = gpflow.train.ScipyOptimizer()
    opt.minimize(m)
    print(m.read_trainables())


    #for testing data
    # num_test = 500
    # test_data_x = np.random.rand(num_test, 1) * 10 - 5
    # test_data_y = step_func(test_data_x)

    # #show the test data
    # fig = plt.figure()
    # ax = fig.add_subplot(111)

    # ax.plot(test_data_x, test_data_y, '*k')

    # plt.show()
    return


if __name__ == '__main__':
    gpflow_nn_ker_test()