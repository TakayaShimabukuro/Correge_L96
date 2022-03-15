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
N = 41
F = 8.0
dt = 0.05
std = 0.001
mu = 0
sigma = 1

# local parameter
logger.info('Prosess Start!!')

file_path = "./q3/result/"

step_2year = 2848
step_1year = 1424
l96 = Model_L96(N, F, dt, std)
plot = Plot_Methods()


# 1. L96を2年分シミュレーションする
logger.info('Prosess 1')
Xn_2year = np.zeros((N, step_2year))
X1_2year = float(F)*np.ones(N)
X1_2year[20] = 1.001*F
Xn_2year, tn_2year = l96.analyze_model(Xn_2year, X1_2year, step_2year)


# 2. 後半1年分を6時間毎に保存する
logger.info('Prosess 2')
Xn_1year = np.zeros((N, step_1year))
Xn_1year = Xn_2year[:, step_1year:step_2year]


# 3. Metsenne Twister法で乱数を生成する
logger.info('Prosess 3')
noise = np.random.normal(loc=mu, scale=sigma, size=step_2year)
noise = list(map(lambda val: val*std, noise))
#plot.graph_noise(noise, file_path)


# 4. 真値にノイズを付加する
logger.info('Prosess 4')
Xn_forcast = np.zeros((N, step_1year))
for i in range(step_1year):
    Xn_forcast[:, i] = list(map(lambda val1, val2: val1 + val2, Xn_1year[:, i], noise))



logger.info('Prosess finish')
