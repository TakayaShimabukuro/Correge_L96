# 外部ライブラリ
from plot import Plot_Methods
from numerical_analysis import Analysis_Methods
from model_l96 import Model_L96
from logging import getLogger, config, DEBUG, basicConfig
import sys
import os
import numpy as np
sys.path.append(os.path.join('..', 'code'))

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
N = 40
F = 8.0
dt = 0.05
n_step = 5000  # 5 step = 1 day, 500 step = 100 day, 5000 step = 1000 day
MODE_SELECT = 3  # 1-Euler, 2-RK2, 3, RK4

# 3年分のシミュレーションを行い、真値の初期値を作成する。
l96 = Model_L96(N, F, dt, n_step)
analyzer = Analysis_Methods()
Xn = analyzer.analyze_model(l96, l96.Xn, l96.X1, N, n_step, MODE_SELECT)

# 初期値Xnをもとに真値Xn[40, :]を取得
n_step_true = 1510  # 300日分取得
Xn_true = np.zeros((N, n_step_true))
Xn_true = analyzer.analyze_model(l96, Xn_true, Xn[:, n_step-1], N, n_step_true, MODE_SELECT)

# 1-100dayに対して、Xn_day[40, day]を初期値とした配列にノイズを付加。
# Xn_dayは[40,1]のXnが100day集まった配列である。


# グラフ出力
file_path = "./q2/result/"
plot = Plot_Methods()
plot.xy_graph_l96(Xn_true, n_step_true, file_path)
logger.info('Prosess finish')
