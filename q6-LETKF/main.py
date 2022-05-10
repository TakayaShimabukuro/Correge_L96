# External Libraries
from logging import getLogger, DEBUG, basicConfig
from matplotlib.pyplot import step
import numpy as np

# Internal Libraries
from plot import Plot_Methods
from model_l96 import Model_L96
from localization import Localization


# PARAMETER
N = 40
F = 8.0
dt = 0.05
delta = 0.05
mu = 0.0
sigma = 1.0
step_2year = 2920
step_t = 1460  # 4step = 1day
num = 100 # member の個数
m = np.arange(20, 20+2*num, 2)
path = "./q6-LETKF/result/"
L = np.zeros((N, N))
L_sigmas = np.arange(1.0, 40.0, 2.0)
spinup = 400
result = []

# DEBUG SETTING
logger = getLogger(__name__)
logger.setLevel(DEBUG)
basicConfig(filename='console.log', level=DEBUG, filemode='w')
np.set_printoptions(threshold=np.inf, suppress=True)

# INSTANCE
plot = Plot_Methods(path)
local = Localization()
l96 = Model_L96(N, F, dt, delta, plot)

# 1. This process is conducted to simulate L96 for 2years
logger.info('--- Start ---')
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

# 4. ETKF

logger.info('Prosess 4')
Xa, Xa_mean, Pb = l96.ETKF(Y, m, step_t)
Xa_RMSE = l96.RMSE(Xa_mean, Xt, step_t)
logger.debug("Xa_RMSE:\n{}".format(Xa_RMSE[0:4]))
Pb_trace = l96.Spread(Pb, step_t)
plot.FuncObTime(t_2year, Xt, Y, Xa_mean, str(len(m))+ "-ETKF")
plot.AnalysisRMSEandTrace(t_2year[:], Xa_RMSE, Pb_trace, "-ETKF")
plot.AnalysisErrCovariance(Pb, "-ETKF")


# 5. LETKF
logger.info('Prosess 5')
for i in range(len(L_sigmas)):
    L = local.get_L(L_sigmas[i])
    plot.Debug(L, "Localization-" + str(L_sigmas[i]))
    Xa, Xa_mean, Pb  = l96.LETKF(Y, m, step_t, L)

    Xa_RMSE = l96.RMSE(Xa_mean, Xt, step_t)
    Pb_trace = l96.Spread(Pb, step_t)
    plot.AnalysisRMSEandTrace(t_2year[:], Xa_RMSE, Pb_trace, "local-" + str(L_sigmas[i]))
    plot.AnalysisErrCovariance(Pb, "local-" + str(L_sigmas[i]))
    result.append(np.mean(Xa_RMSE[spinup:]))

logger.info('--- End ---')