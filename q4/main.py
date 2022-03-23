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

filePath = "./q4/result/"

step_2year = 2848
step_t = 1424
d = [0.00, 0.03, 0.05]
step=[1, 10, 10]
start = [0, 0, 250]
end = [5, 50, 300]
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

for i in range(len(d)):
    for j in range(len(d)):
        # 4. Kalman Filter
        logger.info('Prosess 4')
        Xf, Pf, Xa, Pa = l96.KF(Y, d[i])
        fileName = "funcOfTime-" +  str(d[i]) + "-" + str(start[j]) + "-" + str(end[j]) + ".png"
        XLabel = "time(day)"
        YLabel = "X"
        Title = "EKF, funcOfTime"
        data = [Xt, Y, Xf, Xa, t_2year[0:step_t]]
        params = [start[j], end[j]+1, step[j]]
        names = [filePath, fileName, XLabel, YLabel, Title]
        logger.debug('--- params---')
        logger.debug(params)

        plot.funcOfTime(data, params, names)

    # 5. RMSE & Spread
    '''
    logger.info('Prosess 5')
    Xa_RMSE = l96.RMSE(Xa, Xt, step_t)
    Pa_Spread = l96.Spread(Pa)

    start = 0
    end = start + 175
    fileName = "varianceInfration-" + str(d[i]) + "-" + str(start) + "-" + str(end) + ".png"
    XLabel = "time(day)"
    YLabel = "X"
    Title = "EKF, VarianceInfration"
    data =[Xa_RMSE, Pa_Spread, t_2year[0:step_t]]
    params = [start, end+1]
    names = [filePath, fileName, XLabel, YLabel, Title]
    plot.VarianceInfration(data, params, names)
    '''

logger.info('Prosess finish!!')