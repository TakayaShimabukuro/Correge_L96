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
std = 0.001

# 3年分のシミュレーションを行い、真値の初期値を作成する。
logger.info('Prosess Start!!')
logger.info('Prosess 1')
l96 = Model_L96(N, F, dt, n_step)
analyzer = Analysis_Methods()
Xn = analyzer.analyze_model(l96, l96.Xn, l96.X1, N, n_step, MODE_SELECT)

# 初期値Xnをもとに真値Xn[40, :]を取得
logger.info('Prosess 2')
n_step_true = 1501  # 300日分取得
Xn_true = np.zeros((N, n_step_true))
Xn_true = analyzer.analyze_model(l96, Xn_true, Xn[:, n_step-1], N, n_step_true, MODE_SELECT)


# 1-100stepに対して、観測の初期値となる番号にノイズを付加。
# X1_predsは[40,1]のXnが100step分集まった配列である。
logger.info('Prosess 3')
n_step_noise = 101
X1_preds = []
noise = np.random.normal(loc=0, scale=1, size=N)
for i in range(n_step_noise):
    X1_pred = np.zeros((N, 1))
    X1_pred = Xn_true[:, i] + std * noise
    X1_preds.append(X1_pred)

# X1_predsをそれぞれ100day分時間経過させる。
logger.info('Prosess 4')
n_step_forcast = 501
Xn_pred = np.zeros((N, n_step_forcast))
Xn_preds = []
for i in range(n_step_noise):
    Xn_pred = analyzer.analyze_model(l96, Xn_pred, X1_preds[i], N, n_step_forcast, MODE_SELECT)
    Xn_preds.append(Xn_pred)

# Xn_trueの形式をXn_predsに合わせるため、スライスする。
logger.info('Prosess 5')
Xn_trues = []
for i in range(n_step_noise):
    Xn_trues.append(Xn_true[:, i:i+500])

rmse_times = []
# rmseを導出する。
logger.info('Prosess 6')
for i in range(n_step_noise):
    # i step目のrmseを求める。標準偏差が1ということははじめは1だが、だんだんと増えていく様子が確認できる。
    rmse = analyzer.calculate_RMSE(Xn_trues[i], Xn_preds[i], n_step_noise)
    rmse_times.append(rmse[20])

# グラフ出力
logger.info('Prosess 7')
file_path = "./q2/result/"
plot = Plot_Methods()
plot.xy_graph_rmse(rmse_times, file_path)

logger.info('Prosess finish')
