from logging import getLogger
import numpy as np
from copy import copy
logger = getLogger(__name__)



class Analysis_Methods:
    # オイラー法
    def Euler(self, model, X1_old):
        #logger.info('Euler()')
        X1_new = X1_old + model.dt * model.f_l96(X1_old)
        return X1_new

    # ルンゲクッタ2次
    def RK2(self, model, X1_old):
        #logger.info('RK2()')
        k1 = model.f_l96(X1_old)
        k2 = model.f_l96(X1_old + k1*model.dt)
        X1_new = X1_old + model.dt/2.0 * (k1 + k2)
        return X1_new

    # ルンゲクッタ4次
    def RK4(self, model, X1_old):
        #logger.info('RK4()')
        k1 = model.f_l96(X1_old)
        k2 = model.f_l96(X1_old + k1*model.dt/2.0)
        k3 = model.f_l96(X1_old + k2*model.dt/2.0)
        k4 = model.f_l96(X1_old + k3*model.dt)
        X1_new = X1_old + model.dt/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)
        return X1_new

    # X1[40, 1]を初期値とするstep分のシミュレーションを行う。
    def analyze_model(self, model, Xn, X1, N, step, MODE_SELECT):
        logger.info('get_estimated_data()')
        for i in range(N):
            Xn[i, 0] = copy(X1[i])
        for j in range(1, step):
            if MODE_SELECT==1:
                X1 = self.Euler(model, X1)
            if MODE_SELECT==2:
                X1 = self.RK2(model, X1)
            if MODE_SELECT==3:
                X1 = self.RK4(model, X1)
            Xn[:, j] = X1[:]
        return Xn

