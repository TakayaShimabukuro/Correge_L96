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
d = [0.00, 0.03, 0.05, 0.10, 0.20]

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

step=[1, 10, 10]
start = [0, 0, 250]
end = [5, 50, 300]
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


# 6. Kalman Filter
logger.info('Prosess 6')
Xa_deltas = np.zeros((len(d), step_t))
for i in range(len(d)):
    Xf, Pf, Xa, Pa = l96.KF(Y, d[i])
    Xa_RMSE = l96.RMSE(Xa, Xt, step_t)
    Xa_deltas[i, :] = Xa_RMSE

start = 0
end = 300
step = 60
fileName = "varianceInfration-pickup-" + \
str(d[i]) + "-" + str(start) + "-" + str(end) + ".png"
XLabel = "time(day)"
YLabel = "RMSE"
Title = "EKF, VarianceInfration"
data = [Xa_deltas, t_2year[0:step_t]]
params = [start, end+1, step, len(d)]
names = [filePath, fileName, XLabel, YLabel, Title, d]
plot.VarianceInfrationDelta(data, params, names)

start2 = 0
end2 = 20
step2 = 2.5
fileName2 = "varianceInfration-pickup-" + \
str(d[i]) + "-" + str(start2) + "-" + str(end2) + ".png"
params = [start2, end2+1, step2, len(d), True, 0, 1.4]
names = [filePath, fileName2, XLabel, YLabel, Title, d]
plot.VarianceInfrationDeltaPickUp(data, params, names)

start3 = 220
end3 = 300
step3 = 10
fileName3 = "varianceInfration-pickup-" + \
str(d[i]) + "-" + str(start3) + "-" + str(end3) + ".png"
params = [start3, end3+1, step3, len(d), True, 0, 1.4]
names = [filePath, fileName3, XLabel, YLabel, Title, d]
plot.VarianceInfrationDeltaPickUp(data, params, names)

# 7. getAveRMSE
logger.info('Prosess 7')
l96.showAveRMSE(Xa_deltas, d)

logger.info('Prosess finish!!')
