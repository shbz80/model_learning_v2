from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import argparse
import pprint

from dotmap import DotMap

from bnn_dyn.lt_pred import lt_pred
from bnn_dyn.config import create_config
from blocks_sim import MassSlideWorld
import pickle

logfile = "/home/shahbaz/Research/Software/model_learning/Results/blocks_exp_preprocessed_data_rs_1_gpy.p"

def main(env, ctrl_type, ctrl_args, overrides, logdir):
    ctrl_args = DotMap(**{key: val for (key, val) in ctrl_args})
    cfg = create_config(env, ctrl_type, ctrl_args, overrides, logdir)
    cfg.pprint()

    exp_data = pickle.load(open(logfile, "rb"), encoding='latin1' )
    exp_params = exp_data['exp_params']
    policy_params = exp_params['policy']
    massSlideParams = exp_params['massSlide']
    # policy_params = exp_params['policy']
    massSlideWorld = MassSlideWorld(**massSlideParams)
    massSlideWorld.set_policy(policy_params)
    massSlideWorld.reset()
    policy = massSlideWorld.predict

    if ctrl_type == "MPC":
        cfg.ctrl_cfg.dX = 2
        cfg.ctrl_cfg.dU = 1
        lt_pred_bnn = lt_pred(cfg.ctrl_cfg, policy)
    else:
        raise NotImplementedError()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', type=str, required=True,
                        help='Environment name: select from [cartpole, reacher, pusher, halfcheetah, yumi]')
    parser.add_argument('-ca', '--ctrl_arg', action='append', nargs=2, default=[],
                        help='Controller arguments, see https://github.com/kchua/handful-of-trials#controller-arguments')
    parser.add_argument('-o', '--override', action='append', nargs=2, default=[],
                        help='Override default parameters, see https://github.com/kchua/handful-of-trials#overrides')
    parser.add_argument('-logdir', type=str, default='log',
                        help='Directory to which results will be logged (default: ./log)')
    args = parser.parse_args()

    main(args.env, "MPC", args.ctrl_arg, args.override, args.logdir)
