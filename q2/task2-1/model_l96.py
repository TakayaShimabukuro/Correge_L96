# coding: utf-8
import numpy as np
from numerical_analysis import Analysis_Methods

from logging import getLogger, config
import json

logger = getLogger(__name__)

class Model_L96:
    def __init__(self, N, F, dt, n_step):
        logger.info('__init__()')

        self.N = N
        self.F = F
        self.dt = dt
        self.n_step = n_step
        self.Xn = np.zeros((N, n_step))
    
    def get_estimated_data(self, MODE_SELECT):
        logger.info('get_estimated_data()')

        analyzer = Analysis_Methods()
        if MODE_SELECT==1:
            analyzer.Euler(self.n_step)
        if MODE_SELECT==2:
            analyzer.RK2(self.n_step)
        if MODE_SELECT==3:
            analyzer.RK4(self.n_step)

        return 0

    def f_l96(self, x):
        f = np.zeros((self.N))
        for i in range(2, self.N-1):
            f[i] = (x[i+1]-x[i-2])*x[i-1]-x[i]+self.F

        f[0] = (x[1]-x[self.N-2])*x[self.N-1]-x[0]+self.F
        f[1] = (x[2]-x[self.N-1])*x[0]-x[1]+self.F
        f[self.N-1] = (x[0]-x[self.N-3])*x[self.N-2]-x[self.N-1]+self.F

        return f
    