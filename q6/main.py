# 外部ライブラリ
from logging import getLogger, DEBUG, basicConfig
from matplotlib.pyplot import step
import numpy as np

# 内部ライブラリ
from plot import Plot_Methods
from model_l96 import Model_L96


''' [code sample]
        logger.debug('--- t---')
        logger.debug(t.shape)
        logger.debug('--- X---')
        logger.debug(X.shape)
        logger.debug('--- Y---')
        logger.debug(Y.shape)
        logger.debug('--- X1---')ß
        logger.debug(X1.shape)
        logger.debug('--- X2---')
        logger.debug(X2.shape)ß
        logger.info('------------------------------')
        logger.info('------------------------------')
aaaß
'''

# public parameter
logger = getLogger(__name__)
logger.setLevel(DEBUG)
basicConfig(filename='console.log', level=DEBUG, filemode='w')
np.set_printoptions(threshold=np.inf)
N = 40
F = 8.0
dt = 0.05
delta = 10**-5
mu = 0.0
sigma = 1.0

# local parameter
logger.info('Prosess Start!!')
step_2year = 2920
step_t = 1460 # 4step 1day 
d = 0.0
m = np.arange(20, 1020, 2)
logger.info('-----member : {}------'.format(str(len(m))))
path = "./q6/result/"
plot = Plot_Methods(path)
l96 = Model_L96(N, F, dt, delta, d, plot)

np.set_printoptions(suppress=True)

# 1. L96を2年分シミュレーションする
logger.info('Prosess 1')
Xt_2year = np.zeros((N, step_2year))
Xt1_2year = float(F)*np.ones(N)
Xt1_2year[20] = 1.001*F
Xt_2year, t_2year = l96.analyze_model(Xt_2year, Xt1_2year, step_2year)

# 2. 後半1年分を6時間毎に保存する
logger.info('Prosess 2')
Xt = np.zeros((N, step_t))
Xt = Xt_2year[:, step_t:step_2year]

# 3. 真値にノイズを付加する
logger.info('Prosess 3')
Y = np.zeros((N, step_t))
np.random.seed(0)
for i in range(step_t):
    Y[:, i] = Xt[:, i] + np.random.normal(loc=mu, scale=sigma, size=N)
np.random.seed(None)
# 4. EnKF
logger.info('Prosess 4')
Xa, Xa_mean, Pa = l96.EnKF_PO(Y, m, step_t)

# 5. FuncObTime
logger.info('Prosess 5')
plot.FuncObTime(t_2year, Xt, Y, Xa_mean, str(len(m)))

# 6. RMSE, Trace
logger.info('Prosess 6')
Xa_RMSE = l96.RMSE(Xa_mean, Xt, step_t)
Pa_trace = l96.Spread(Pa, step_t)

#6. AnalysisRMSEandTrace
plot.AnalysisRMSEandTrace(t_2year[:], Xa_RMSE, Pa_trace, str(len(m)))
plot.AnalysisErrCovariance(Pa)


logger.info('Prosess Finish!!')
