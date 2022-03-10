# 外部ライブラリ
from logging import getLogger, config, DEBUG, basicConfig
import sys
import os
import numpy as np
sys.path.append(os.path.join('..', 'code'))

# 内部ライブラリ
from plot import Plot_Methods
from numerical_analysis import Analysis_Methods
from model_l96 import Model_L96
# 内部ライブラリ

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
basicConfig(filename='console.log', level=DEBUG)


# パラメータ設定
N = 41
F = 8.0
dt = 0.05
n_step = 5000  # 5 step = 1 day, 500 step = 100 day, 5000 step = 1000 day
MODE_SELECT = 3  # 1-Euler, 2-RK2, 3, RK4


# 3年分のシミュレーションを行い、真値の初期値を作成する。
logger.info('Prosess Start!!')
logger.info('Prosess 1')
l96 = Model_L96(N, F, dt, n_step)
analyzer = Analysis_Methods()
Xn = analyzer.analyze_model(l96, l96.Xn, l96.X1, N, n_step, MODE_SELECT)


# 初期値Xnをもとに真値Xn[40, :]を取得
logger.info('Prosess 2')
n_step_true = 3000
Xn_true = analyzer.analyze_model(l96, np.zeros(
    (N, n_step_true)), Xn[:, n_step-1], N, n_step_true, MODE_SELECT)


# 真値の1-1000stepをそれぞれ初期値とした配列を作成
logger.info('Prosess 3')
n_step_forcast_init = 1000
X1_forcasts = []
for i in range(n_step_forcast_init):
    X1_forcast = analyzer.add_noise(Xn_true[:, i], N)
    X1_forcasts.append(X1_forcast)


# 上記をそれぞれ100step分時間経過させる。[40, 100]が1000個
logger.info('Prosess 4')
n_step_forcast = 100
Xn_forcasts = []
for i in range(n_step_forcast_init):
    Xn_forcasts.append(
        analyzer.analyze_model(l96, np.zeros(
            (N, n_step_forcast)), X1_forcasts[i], N, n_step_forcast, MODE_SELECT)
    )


# Xn_truesをスライス
logger.info('Prosess 5')
Xn_trues = []
for i in range(n_step_forcast_init):
    Xn_trues.append(Xn_true[:, i:i+n_step_forcast])


# rmseを導出する。
logger.info('Prosess 6')
rmse_times = []
for i in range(n_step_forcast_init):
    # i step目のrmseを求める。標準偏差が1ということははじめは1だが、だんだんと増えていく様子が確認できる。
    rmse = analyzer.calculate_RMSE(Xn_trues[i], Xn_forcasts[i], n_step_forcast)
    # logger.debug(rmse[0])
    rmse_mean = np.mean(rmse)
    rmse_times.append(rmse_mean)


# グラフ出力
logger.info('Prosess 7')
file_path = "./q2/result/"
plot = Plot_Methods()
plot.xy_graph_rmse(rmse_times, file_path)

logger.info('Prosess finish')
