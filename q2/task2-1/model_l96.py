# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from sklearn.metrics import mean_squared_error
from logging import getLogger, config
import json

#logの初期設定
log_config = open('./log_config.json', 'r')
config.dictConfig(json.load(log_config))
logger = getLogger(__name__)


class Model_L96:
    def __init__(self, N, F, dt, n_step):
        self.N = N
        self.F = F
        self.dt = dt
        self.n_step = n_step
        self.Xn = np.zeros((N, n_step))
        print("Model_L96 : __init__ ")

    def get_estimated_data(self):
        X = float(self.F)*np.ones(self.N)
        X[20] = 1.001*self.F
        #self.Xn = self.create_table(self.N, self.DAYS, X, self.Xn)