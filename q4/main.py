# 外部ライブラリ
from logging import getLogger, DEBUG, basicConfig
import numpy as np

# 内部ライブラリ
from plot import Plot_Methods
from model_l96 import Model_L96


''' [code sample]
logger.info('Process Start!')
logger.debug('debug')
logger.info('info')
logger.warning('warning')
logger.error('error')
logger.info('Process End!')
'''

# logの初期設定
logger = getLogger(__name__)
logger.setLevel(DEBUG)
basicConfig(filename='console.log', level=DEBUG, filemode='w')


# public parameter
N = 40
F = 8.0
dt = 0.05

mu = 0
sigma = 1

# local parameter
logger.info('Prosess Start!!')

file_path = "./q4/result/"

step_2year = 2848
step_t = 1424
l96 = Model_L96(N, F, dt)
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
Xf, Pf, Xa, Pa = l96.KF(Y)


for i in range(5):
    if i%1 == 0:
        logger.debug(Xf[:, i])
        
logger.info('')
logger.info('')
logger.info('')

for i in range(5):
    if i%1 == 0:
        logger.debug(Xa[:, i])


# 5. plot data
logger.info('Prosess 5')
plot.funcOfTime(Xt, Y, Xf, Xa, t_2year[0:step_t], file_path)

# 6. plot RMSE
logger.info('Prosess 6')
#rmse1 = l96.RMSE(Xf, Y)
#rmse2 = l96.RMSE(Xa, Y)

#logger.debug(Xf)
#logger.debug(len(t_2year[0:step_t]))
#plot.RMSE(rmse1, rmse2, t_2year[0:step_t], file_path)


logger.info('Prosess finish!!')