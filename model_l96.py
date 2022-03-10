# coding: utf-8
import numpy as np
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
        self.X1= float(F)*np.ones(N)
        self.X1[20] = 1.001*F
    
    def f_l96(self, x):
        #logger.info('f_l96()')
        
        f = np.zeros((self.N))
        for i in range(2, self.N-1):
            f[i] = (x[i+1]-x[i-2])*x[i-1]-x[i]+self.F

        f[0] = (x[1]-x[self.N-2])*x[self.N-1]-x[0]+self.F
        f[1] = (x[2]-x[self.N-1])*x[0]-x[1]+self.F
        f[self.N-1] = (x[0]-x[self.N-3])*x[self.N-2]-x[self.N-1]+self.F

        return f
    