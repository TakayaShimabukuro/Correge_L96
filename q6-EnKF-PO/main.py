# EXTERNAL LIBRARIES
from logging import getLogger, DEBUG, basicConfig
import numpy as np
import tqdm
import threading
import queue
import time

# INTERNAL LIBRARIES
from plot import Plot_Methods
from model_l96 import Model_L96
from localization import Localization


# PARAMETER
N = 40
F = 8.0
dt = 0.05
delta = 0.1
mu = 0.0
sigma = 1.0
step_2year = 2920
step_t = 1460  # 4step = 1day
ensamble_size = 40
ensambles = np.arange(0, ensamble_size, 1)
path = "./q6-EnKF-PO/result/"
L_sigmas = np.arange(1.0, N, 2.0)
spinup = 80
Xas_RMSE_mean = []

Ï
# DEBUG SETTING
logger = getLogger(__name__)
logger.setLevel(DEBUG)
basicConfig(filename='console.log', level=DEBUG, filemode='w')
np.set_printoptions(threshold=np.inf, suppress=True)


# INSTANCE
plot = Plot_Methods(path)
local = Localization(N)
l96 = Model_L96(N, F, dt, delta, plot)

# METHODS
def process4_multi(i, Xt, Y, q):
    L = local.get_L(L_sigmas[i])
    Xa, Xa_mean, Pa  = l96.EnKF_PO(Y, ensambles, ensamble_size, step_t, L)
    Xa_RMSE = l96.RMSE(Xa_mean, Xt, step_t)
    Pa_trace = l96.Spread(Pa, step_t)
    q.put(np.mean(Xa_RMSE[spinup:]))

# PROCESS
if __name__ == '__main__':
    start = time.time()
    logger.info('--- PROCESS 1 ')
    logger.info(" N = %d ---", N)
    logger.info(" ensamble_size = %d ", ensamble_size)
    logger.info(" spinup = %d ", spinup)
    Xt_2year = np.zeros((N, step_2year))
    Xt1_2year = float(F)*np.ones(N)
    Xt1_2year[20] = 1.001*F
    Xt_2year, t_2year = l96.analyze_model(Xt_2year, Xt1_2year, step_2year)

    logger.info('--- PROCESS 2 ---')
    Xt = np.zeros((N, step_t))
    Xt = Xt_2year[:, step_t:step_2year]

    logger.info('--- PROCESS 3 ---')
    Y = np.zeros((N, step_t))
    np.random.seed(0)
    for i in range(step_t):
        Y[:, i] = Xt[:, i] + np.random.normal(loc=mu, scale=sigma, size=N)
    np.random.seed(None)

    logger.info('--- PROCESS 4 ---')
    for i in tqdm.tqdm(range(len(L_sigmas))):
        print(threading.active_count())
        globals()[f"q{i}"] = queue.Queue()
        globals()[f"t{i}"] = threading.Thread(target=process4_multi, args=(i, Xt, Y, globals()[f"q{i}"]))

        globals()[f"t{i}"].start()
        Xas_RMSE_mean.append(globals()[f"q{i}"].get())
        logger.debug("--- L_sigma = %d, Xa_RMSE = %f---", L_sigmas[i], Xas_RMSE_mean[i])

    logger.info('--- PROCESS 5 ---')
    plot.TimeMeanRMSE(L_sigmas, Xas_RMSE_mean)
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")