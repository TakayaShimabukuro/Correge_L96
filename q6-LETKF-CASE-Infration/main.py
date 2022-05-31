# EXTERNAL LIBRARIES
from cmath import log
from logging import getLogger, DEBUG, basicConfig
import numpy as np
from sympy import false
import tqdm
import time

# INTERNAL LIBRARIES
from plot import Plot_Methods
from model_l96 import Model_L96
from localization import Localization


# PARAMETER
N = 40
F = 8.0
dt = 0.05
infration = np.arange(0.01, 0.1, 0.01)
step_2year = 2920
step_t = 1460  # 4step = 1day
ensamble_size = 8
path = "./q6-LETKF-CASE-Infration/result/"
title = "Lecture6-LETKF-case-infration"
L_sigmas = np.arange(1.0, 11.0, 1.0)
spinup = 80

Xt_2year = np.zeros((N, step_2year))
Xt1_2year = float(F)*np.ones(N)
Xt = np.zeros((N, step_t))
Y = np.zeros((N, step_t))

# DEBUG SETTING
logger = getLogger(__name__)
logger.setLevel(DEBUG)
basicConfig(filename='console.log', level=DEBUG, filemode='w')
np.set_printoptions(threshold=np.inf, suppress=True)

Xas_case=np.zeros([len(infration),len(L_sigmas)])

for inf in tqdm.tqdm(range(len(infration))):
    # INSTANCE
    plot = Plot_Methods(path, title)
    local = Localization(N)
    l96 = Model_L96(N, F, dt, infration[inf], plot)

    # PROCESS
    if __name__ == '__main__':
        logger.info('--- PARAMETER ---')
        logger.debug(" N : %d ", N)
        logger.debug(" ensamble_size : %d ", ensamble_size)
        logger.debug(f' infration[inf]: {infration[inf]:.0%}')
        logger.debug(" spinup : %d day", (spinup/4))
        logger.debug(Xas_case.shape)
        logger.info('--- DEBUG ---')

        delate_queue1 = l96.get_deleate_queue(2)
        delate_queue2 = l96.get_deleate_queue(1)

        #Process 1
        
        Xt1_2year[20] = 1.001*F
        Xt_2year, t_2year = l96.analyze_model(Xt_2year, Xt1_2year, step_2year)

        #Process 2
        Xt = Xt_2year[:, step_t:step_2year]
        
        #Process 3
        np.random.seed(2)
        for i in tqdm.tqdm(range(step_t), leave=False):
            Y[:, i] = Xt[:, i] + np.random.normal(loc=0.0, scale=1.0, size=N)
        np.random.seed(None)

        #Process 4
        
        for j in tqdm.tqdm(range(len(delate_queue1))):
            p = 40 - len(delate_queue1[j])
            if(p == 20):
                Xas_RMSE_mean = []
                for i in tqdm.tqdm(range(len(L_sigmas)), leave=False):
                    Xa, Xa_mean, Pb = l96.LETKF(Y, ensamble_size, step_t, L_sigmas[i], delate_queue1[j])
                    Xa_RMSE = l96.RMSE(Xa_mean, Xt, step_t)
                    Xas_RMSE_mean.append(np.mean(Xa_RMSE[spinup:]))
                    Xas_case[inf, i] = Xas_RMSE_mean[i]
                    logger.debug("Obs = %d, L_sigmas = %d, Xa_RMSE = %f", p, L_sigmas[i], Xas_RMSE_mean[i])
                    logger.debug(Xas_case[0:5, 0:5])


#Process 5
plot.RMSEOfInfandL(Xas_case[:, :], L_sigmas, infration)