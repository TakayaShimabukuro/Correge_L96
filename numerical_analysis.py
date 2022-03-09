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
    
    # 初期値導出
    def get_init_condition(self, model):
        logger.info('set_init_array()')

        X1= float(model.F)*np.ones(model.N)
        X1[20] = 1.001*model.F
        return X1
    
    # 観測値導出
    def get_estimated_data(self, model, X1, MODE_SELECT):
        logger.info('get_estimated_data()')
        for i in range(model.N):
            model.Xn[i, 0] = copy(X1[i])
        for j in range(1, model.n_step):
            if MODE_SELECT==1:
                X1 = self.Euler(model, X1)
            if MODE_SELECT==2:
                X1 = self.RK2(model, X1)
            if MODE_SELECT==3:
                X1 = self.RK4(model, X1)
            model.Xn[:, j] = X1[:]
            model.t[j] = model.dt*j
        return model.Xn

    # 実行メソッド
    def analyze_model(self, model, MODE_SELECT):
        logger.info('analyze_model()')
        try:
            X1 = self.get_init_condition(model)
            Xn = self.get_estimated_data(model, X1, MODE_SELECT)
            return Xn

        except Exception as e:
            logger.error(e)
            return 0

        
    
    def analyze_models(self, model, MODE_SELECT, F_list):
        logger.info('analyze_models()')
        
        try:
            Xn_list = []
            for i, val in enumerate(F_list):
                model.F = F_list[i]
                
                X1 = self.get_init_condition(model)
                Xn = self.get_estimated_data(model, X1, MODE_SELECT)
                Xn_list.append(Xn)
            return Xn_list

        except Exception as e:
            logger.error(e)
            return 0
        

