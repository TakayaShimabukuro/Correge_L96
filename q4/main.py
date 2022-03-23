# 外部ライブラリ
from logging import getLogger, DEBUG, basicConfig
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

# logの初期設定
logger = getLogger(__name__)
logger.setLevel(DEBUG)
basicConfig(filename='console.log', level=DEBUG, filemode='w')


# public parameter
N = 40
F = 8.0
dt = 0.05
delta = 10**-5

mu = 0
sigma = 1

# local parameter
logger.info('Prosess Start!!')

file_path = "./q4/result/"

step_2year = 2848
step_t = 1424
l96 = Model_L96(N, F, dt, delta)
plot = Plot_Methods()

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
for i in range(step_t):
    Y[:, i] = Xt[:, i] + np.random.normal(loc=mu, scale=sigma, size=N)

# 4. Kalman Filter
logger.info('Prosess 4')
day = 50
No = 10
Xf, Pf, Xa, Pa = l96.KF(Y)
#plot.funcOfTime(Xt, Y, Xf, Xa, t_2year[0:step_t], day, No, file_path)

# 5. RMSE & Spread
logger.info('Prosess 5')
Xf_RMSE = l96.RMSE(Xf, Xt, step_t)
Xa_RMSE = l96.RMSE(Xa, Xt, step_t)
Pf_Spread = l96.Spread(Pf)
Pa_Spread = l96.Spread(Pa)
plot.VarianceInfration(Xf_RMSE, Pf_Spread, t_2year[0:step_t], 175, file_path)


logger.info('Prosess finish!!')