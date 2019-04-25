from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from dotmap import DotMap

from dmbrl.misc.DotmapUtils import get_required_argument
from dmbrl.modeling.layers import FC
import dmbrl.env


class BlocksConfigModule:
    ENV_NAME = "blocks"
    TASK_HORIZON = 40
    NTRAIN_ITERS = 1
    NROLLOUTS_PER_ITER = 1
    PLAN_HOR = 40
    MODEL_IN, MODEL_OUT = 3, 2
    GP_NINDUCING_POINTS = 10

    def __init__(self):
        self.ENV = None
        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        self.SESS = tf.Session(config=cfg)
        self.NN_TRAIN_CFG = {"epochs": 5}
        self.OPT_CFG = {
            "Random": {
                "popsize": 2500
            },
            "CEM": {
                "popsize": 500,
                "num_elites": 50,
                "max_iters": 5,
                "alpha": 0.1
            }
        }
        self.dX = 2
        self.dU = 1

    def nn_constructor(self, model_init_cfg):
        model = get_required_argument(model_init_cfg, "model_class", "Must provide model class")(DotMap(
            name="model", num_networks=get_required_argument(model_init_cfg, "num_nets", "Must provide ensemble size"),
            sess=self.SESS, load_model=model_init_cfg.get("load_model", False),
            model_dir=model_init_cfg.get("model_dir", None)
        ))
        if not model_init_cfg.get("load_model", False):
            model.add(FC(200, input_dim=self.MODEL_IN, activation="swish", weight_decay=0.000025))
            model.add(FC(200, activation="swish", weight_decay=0.00005))
            model.add(FC(self.MODEL_OUT, weight_decay=0.0001))
        model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})
        return model



CONFIG_MODULE = BlocksConfigModule
