#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
yacs config default parameters
"""

from yacs.config import CfgNode as CN

_C = CN()

_C.SEED = 1

_C.NUMERAI = CN()
_C.NUMERAI.ID = "YOUR NUMERAI ID"
_C.NUMERAI.KEY = "YOUR NUMERAI KEY"

_C.MODEL = CN()
_C.MODEL.TYPE = "mlp" # "mlp", "gbdt"

_C.MODEL.ERABOOST = True
_C.MODEL.ERABOOST_ITER = 10
_C.MODEL.TIME_SERIES_CV = False
_C.MODEL.CV_ITER = 5
_C.MODEL.DIM_OUT = 1

# parameter for mlp
_C.MODEL.NUM_STAGE = 8
_C.MODEL.NUM_HIDDEN = 1
_C.MODEL.DIM_HIDDEN = 1024
_C.MODEL.DIM_HIDDEN_FACT = 1 / 2
_C.MODEL.P_DROPOUT = 0.1
_C.MODEL.PRED_LOSS = "bce" #"mse", "l1", "bce", "corr"
_C.MODEL.REG_TYPE = "l1" #"l1", "l2"
_C.MODEL.REG_WEIGHT = 0.000001
_C.MODEL.ACTIVATION = "leaky-relu" #"relu", "leaky-relu", "sigmoid"
_C.MODEL.RESIDUAL = False

# parameter for gbdt
_C.MODEL.GBDT = CN()
_C.MODEL.GBDT.LR = 0.01
_C.MODEL.GBDT.MAX_DEPTH = 5
_C.MODEL.GBDT.NUM_LEAVES = 2 ** 5
_C.MODEL.GBDT.N_ESTIMATORS = 2000
_C.MODEL.GBDT.REG_ALPHA = 0.0
_C.MODEL.GBDT.REG_LAMBDA = 0.0
        

_C.FE = CN()
_C.FE.APPLY = False
_C.FE.FREEZE = True
_C.FE.TYPE = "autoencoder" #"autoencoder"
_C.FE.NUM_HIDDEN = 2
_C.FE.DIM_HIDDEN = 128
_C.FE.DIM_HIDDEN_FACT = 1 / 2
_C.FE.DIM_OUT = 32
_C.FE.P_DROPOUT = 0.1
_C.FE.PRED_LOSS = "bce" #"mse", "l1", "bce"
_C.FE.REG_TYPE = "l1" #"l1", "l2"
_C.FE.REG_WEIGHT = 0.000001

_C.SOLVER = CN()
_C.SOLVER.NUM_WORKERS = 4
_C.SOLVER.OPTIMIZER = "radam"  #"sgd", "radam"
_C.SOLVER.LR = 1e-3
_C.SOLVER.ITER = 1000
_C.SOLVER.DECAY_FACTOR = 0.1
_C.SOLVER.AUG = None

_C.OUTPUT_DIR = "./output/"
_C.DATASET_DIR = "./dataset/"

_C.MDA = CN()
_C.MDA.APPLY = False
_C.MDA.MODEL = "weights/model.pkl"
_C.MDA.FEATURE_NUM = 300
_C.MDA.TEST_SET = "val"

_C.NEUTRALIZE_COEF = 0.25  #currently not functional

_C.LOAD_TUNED = False