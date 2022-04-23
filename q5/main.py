# External Libraries
from logging import getLogger, DEBUG, basicConfig
from matplotlib.pyplot import step
import numpy as np

# Internal Libraries
from plot import Plot_Methods
from model_l96 import Model_L96

# PARAMETER
N = 40
F = 8.0
dt = 0.05
delta = 10**-5
mu = 0.0
sigma = 1.0
step_2year = 2920
step_t = 1460  # 4step = 1day
m = np.arange(20, 1020, 2)
d = [0.05]
B_step = np.arange(0.05, 0.625, 0.025)
path = "./q5/result/"
path_debug = "./q5/Debug/"
Xfs = []
Pfs = []
Xas = []
Pas = []
Xas_RMSE = []
Pas_Spread = []
Xa_RMSE_aves = []
Xa_RMSE_aves_case1 = []
Xa_RMSE_aves_case2 = []
Xa_3DVAR_best1 = []
Xa_3DVAR_best2 = []
Xa_KF_best1 = []
Xa_KF_best2 = []

# DEBUG SETTINGß
logger = getLogger(__name__)
logger.setLevel(DEBUG)
basicConfig(filename='console.log', level=DEBUG, filemode='w')
np.set_printoptions(threshold=np.inf)
np.set_printoptions(threshold=np.inf, suppress=True)

# INSTANCE
plot = Plot_Methods(path, path_debug)
l96 = Model_L96(N, F, dt, delta, d)

# 0. This process is conducted to get delete queue
logger.info('Prosess 1')
delate_queue1 = l96.get_deleate_queue(2)
delate_queue2 = l96.get_deleate_queue(1)
spinup = 400

# 1. This process is conducted to simulate L96 for 2years
logger.info('Prosess 1')
Xt_2year = np.zeros((N, step_2year))
Xt1_2year = float(F)*np.ones(N)
Xt1_2year[20] = 1.001*F
Xt_2year, t_2year = l96.analyze_model(Xt_2year, Xt1_2year, step_2year)

# 2. This process is conducted to save the latter half of Xt_2year
logger.info('Prosess 2')
Xt = np.zeros((N, step_t))
Xt = Xt_2year[:, step_t:step_2year]

# 3. set seed of noise and add noise
logger.info('Prosess 3')
Y = np.zeros((N, step_t))
np.random.seed(0)
for i in range(step_t):
    noise = np.random.normal(loc=mu, scale=sigma, size=N)
    Y[:, i] = Xt[:, i] + noise
np.random.seed(None)

# 4.This process is conducted to analize 3DVAR.
logger.info('Prosess 4')
for i in range(len(B_step)):
    B = np.diag([B_step[i]]*N)
    Xa = l96.analyze_3DVAR(Y, B, step_t)
    Xas.append(Xa)

# 5. This process is conducted to get RMSE.
logger.info('Prosess 5')
np.set_printoptions(threshold=np.inf)
for i in range(len(B_step)):
    Xa_RMSE = l96.RMSE(Xas[i], Xt, step_t)
    Xa_RMSE_Tstep_mean = np.mean(Xa_RMSE[spinup:])
    Xa_RMSE_aves.append(Xa_RMSE_Tstep_mean)
    #logger.debug("B={%f}, ave RMSE(Xa)={%f}", B_step[i], Xa_RMSE_Tstep_mean)

# 6. This process is conducted to plot RMSE caluculated mean of time.
logger.info('Prosess 6')
plot.TimeMeanRMSE(B_step, Xa_RMSE_aves, str(spinup))

# 7. provided that H is incomplete(data number -> Max:40, min:20), 
# This process is conducted to get RMSE caluculated mean of time.
logger.info('Prosess 7')
for i in range(len(delate_queue1)):
    logger.debug(str(i+1) + "/" + str(len(delate_queue1)+1))
    Xa_3DVAR_case1 = l96.analyze_3DVAR_case(
        Y, np.diag([0.225]*N), step_t, delate_queue1[i])
    Xa_3DVAR_case2 = l96.analyze_3DVAR_case(
        Y, np.diag([0.225]*N), step_t, delate_queue2[i])
    Xa_3DVAR_best1.append(np.mean(l96.RMSE(Xa_3DVAR_case1, Xt, step_t)[400:]))
    Xa_3DVAR_best2.append(np.mean(l96.RMSE(Xa_3DVAR_case2, Xt, step_t)[400:]))

    Xa_KF_case1 = l96.KF(Y, 0.15, delate_queue1[i])
    Xa_KF_case2 = l96.KF(Y, 0.15, delate_queue2[i])
    Xa_KF_best1.append(np.mean(l96.RMSE(Xa_KF_case1, Xt, step_t)[400:]))
    Xa_KF_best2.append(np.mean(l96.RMSE(Xa_KF_case2, Xt, step_t)[400:]))

# 8. This process is conducted to plot RMSE(H is incomplete) caluculated mean of time.
logger.info('Prosess 8')
plot.ComparisonOfEKFand3DVAR(Xa_3DVAR_best1, Xa_KF_best1, "case1")
plot.ComparisonOfEKFand3DVAR(Xa_3DVAR_best2, Xa_KF_best2, "case2")
