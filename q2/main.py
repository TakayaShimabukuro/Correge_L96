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


# パラメータ設定
N = 41
F = 8.0
dt = 0.05
std = 0.001

logger.info('Prosess Start!!')
# 3年分のシミュレーションを行い、真値の初期値を作成する。
logger.info('Prosess 1')
l96 = Model_L96(N, F, dt, std)

n_step_init = 5000 
Xn_init = np.zeros((N, n_step_init))
X1_init= float(F)*np.ones(N)
X1_init[20] = 1.001*F

Xn_init, tn_init= l96.analyze_model(Xn_init, X1_init, n_step_init)


# 初期値Xnをもとに真値Xn[40, :]を取得
logger.info('Prosess 2')

n_step_true = 4000
Xn_true = np.zeros((N, n_step_true))
X1_true = Xn_init[:, n_step_init-1]

Xn_true, tn_true = l96.analyze_model(Xn_true, X1_true, n_step_true)


# 真値の1-1000stepをそれぞれ初期値とし,　そこにノイズを加え、時間経過
logger.info('Prosess 3')

n_step_forcast_init = 1000
n_step_forcast = 9
Xn_forcasts = []
for i in range(n_step_forcast_init):
    Xn_forcast = np.zeros((N, n_step_forcast))
    X1_forcast = l96.add_noise(Xn_true[:, i])

    Xn_forcast, tn_forcast = l96.analyze_model(Xn_forcast, X1_forcast, n_step_forcast)
    Xn_forcasts.append(Xn_forcast)


# Xn_truesをスライス
logger.info('Prosess 4')
Xn_trues = []
for i in range(n_step_forcast_init):
    Xn_trues.append(Xn_true[:, i:i+n_step_forcast])


# rmseを導出する。
logger.info('Prosess 5')
#rmse = l96.get_RMSE(Xn_trues, Xn_forcasts, n_step_forcast)

num_list = np.arange(0, n_step_forcast_init, step=100)
rmse_list = []
for i in range(len(num_list)):
    rmse_list.append(l96.get_RMSE_solo(Xn_trues, Xn_forcasts, n_step_forcast, num_list[i]))

# グラフ出力
logger.info('Prosess 6')
file_path = "./q2/result/"
plot = Plot_Methods()
for i in range(len(num_list)):
    plot.xy_graph_rmse_list(rmse_list, file_path, tn_forcast, std, "list")

logger.info('Prosess finish')
