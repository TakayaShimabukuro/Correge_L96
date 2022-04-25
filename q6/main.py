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
delta = 10**-5
mu = 0.0
sigma = 1.0
step_2year = 2920
step_t = 1460  # 4step = 1day
m = np.arange(20, 320, 2)
path = "./q6/result/"

# DEBUG SETTING
logger = getLogger(__name__)
logger.setLevel(DEBUG)
basicConfig(filename='console.log', level=DEBUG, filemode='w')
np.set_printoptions(threshold=np.inf, suppress=True)

# INSTANCE
plot = Plot_Methods(path)
local = Localization()
l96 = Model_L96(N, F, dt, delta, plot, local)

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

# 4. This process is conducted to analyze using EnKF and plot thier data.
logger.info('Prosess 4')
Xa, Xa_mean, Pa = l96.EnKF_PO(Y, m, step_t, False)
#plot.FuncObTime(t_2year, Xt, Y, Xa_mean, str(len(m)))

# 5. This process is conducted to get Xa RMSE and Pb Trace and plot thier data.
logger.info('Prosess 5')
Xa_RMSE = l96.RMSE(Xa_mean, Xt, step_t)
Pa_trace = l96.Spread(Pa, step_t)
plot.AnalysisRMSEandTrace(t_2year[:], Xa_RMSE, Pa_trace, "-all")
plot.AnalysisErrCovariance(Pa, "-all")

# 6. This process is conducted to locally analyze using EnKF and plot thier data.
logger.info('Prosess 6')
Xa_local, Xa_mean_local, Pa_local = l96.EnKF_PO(Y, m, step_t, True)
Xa_RMSE_local = l96.RMSE(Xa_mean_local, Xt, step_t)
Pa_trace_local = l96.Spread(Pa_local, step_t)
plot.AnalysisRMSEandTrace(t_2year[:], Xa_RMSE_local, Pa_trace_local, "-local")
plot.AnalysisErrCovariance(Pa_local, "-local")
