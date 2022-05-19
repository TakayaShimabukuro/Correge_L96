# EXTERNAL LIBRARIES
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
infration = 0.03
step_2year = 2920
step_t = 1460  # 4step = 1day
ensamble_size = 20
path = "./q6-LETKF/result/"
title = "Lecture6-LETKF"
L_sigmas = np.arange(1.0, 16, 2.0)
spinup = 80
Xas_RMSE_mean = []
Xt_2year = np.zeros((N, step_2year))
Xt1_2year = float(F)*np.ones(N)
Xt = np.zeros((N, step_t))
Y = np.zeros((N, step_t))

# DEBUG SETTING
logger = getLogger(__name__)
logger.setLevel(DEBUG)
basicConfig(filename='console.log', level=DEBUG, filemode='w')
np.set_printoptions(threshold=np.inf, suppress=True)


# INSTANCE
plot = Plot_Methods(path, title)
local = Localization(N)
l96 = Model_L96(N, F, dt, infration, plot)


# PROCESS
if __name__ == '__main__':
    logger.info('--- PARAMETER ---')
    logger.debug(" N : %d ", N)
    logger.debug(" ensamble_size : %d ", ensamble_size)
    logger.debug(f' infration: {infration:.0%}')
    logger.debug(" spinup : %d day", (spinup/4))
    logger.info('--- DEBUG ---')

    #Process 1
    start = time.time()
    Xt1_2year[20] = 1.001*F
    Xt_2year, t_2year = l96.analyze_model(Xt_2year, Xt1_2year, step_2year)

    #Process 2
    Xt = Xt_2year[:, step_t:step_2year]
    
    #Process 3
    np.random.seed(0)
    for i in tqdm.tqdm(range(step_t), leave=False):
        Y[:, i] = Xt[:, i] + np.random.normal(loc=0.0, scale=1.0, size=N)
    np.random.seed(None)

    #Process 4
    for i in tqdm.tqdm(range(len(L_sigmas)), leave=False):
        L = local.get_L(L_sigmas[i])
        Xa, Xa_mean, Pb  = l96.LETKF(Y, ensamble_size, step_t, L)
        Xa_RMSE = l96.RMSE(Xa_mean, Xt, step_t)
        Xas_RMSE_mean.append(np.mean(Xa_RMSE[spinup:]))
        logger.debug(" L_sigmas = %d, Xa_RMSE = %f", L_sigmas[i], Xas_RMSE_mean[i])

        Pb_trace = l96.Spread(Pb, step_t)
        plot.AnalysisRMSEandTrace(t_2year[:], Xa_RMSE, Pb_trace, str(L_sigmas[i]))

    #Process 5
    plot.TimeMeanRMSE(L_sigmas, Xas_RMSE_mean)
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")