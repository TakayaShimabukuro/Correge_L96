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
        logger.debug('--- X1---')
        logger.debug(X1.shape)
        logger.debug('--- X2---')
        logger.debug(X2.shape)
        logger.info('------------------------------')
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
step_2year = 2848
step_t = 1424
#d = np.arange(0, 0.20, 0.025)
d = [0.00]
B = np.arange(0.05, 0.625, 0.025)
m = np.arange(10, 30, 5)
path = "./q6/result/"
plot = Plot_Methods(path)
l96 = Model_L96(N, F, dt, delta, d, plot)
Xfs = []
Pfs = []
Xas = []
Pas = []
Xas_RMSE = []
Pas_Spread = []
Xa_RMSE_aves = []

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

# 4. EnKF
logger.info('Prosess 4')
Xa = l96.EnKF_PO(Y, m)

logger.info('Prosess Finish!!')
