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
# public parameter
logger = getLogger(__name__)
logger.setLevel(DEBUG)
basicConfig(filename='console.log', level=DEBUG, filemode='w')
N = 40
F = 8.0
dt = 0.05
delta = 10**-5
mu = 0
sigma = 1

# local parameter
logger.info('Prosess Start!!')
step_2year = 2848
step_t = 1424
#d = np.arange(0, 0.20, 0.025)
d = [0.00, 0.03, 0.05, 0.1, 0.2, 0.75, 0.15, 0.225]
path = "./q4/result/"
l96 = Model_L96(N, F, dt, delta, d)
plot = Plot_Methods(path)
Xfs = []
Pfs = []
Xas = []
Pas = []
Xas_RMSE = []
Pas_Spread = []

FLG = True
while FLG:
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
    for i in range(len(d)):
        Xf, Pf, Xa, Pa = l96.KF(Y, d[i])
        Xfs.append(Xf)
        Pfs.append(Pf)
        Xas.append(Xa)
        Pas.append(Pa)

    # 5. Variance Inflation
    logger.info('Prosess 5')
    for i in range(len(d)):
        Xas_RMSE.append(l96.RMSE(Xas[i], Xt, step_t))
        Pas_Spread.append(l96.Spread(Pas[i]))
    #plot.VarianceInfration(d, t_2year, Xas_RMSE, Pas_Spread)

    # 6. First Variable X(1) as a func. of time
    logger.info('Prosess 6')
    #plot.X1asFuncOfTime(d, t_2year, Xt, Y, Xfs, Xas)

    # 7. Analysis RMSE
    logger.info('Prosess 7')
    #plot.AnalysisRMSE(d, t_2year, Xas_RMSE)

    # 8. Sensitivity to Infl. Factor
    logger.info('Prosess 8')
    rmse_aves = l96.get_RMSE_Ave(d, Xas_RMSE)
    #if rmse_aves[2] < 0.4:
    FLG = False
    #plot.InflationRatio(d, rmse_aves)

    # 9. Analysis Error Covariance Pa
    logger.info('Prosess 9')
    #plot.AnalysisErrCovariance(d, Pas)

logger.info('Prosess Finish!!')
