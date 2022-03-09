# 外部ライブラリ
from plot import Plot_Methods
from numerical_analysis import Analysis_Methods
from model_l96 import Model_L96
from logging import getLogger, config, DEBUG, basicConfig
import sys
import os

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
F = 1.0
dt = 0.05
n_step = 3000
MODE_SELECT = 3  # 1-Euler, 2-RK2, 3, RK4
F_list = [0.1, 0.5, 1.0, 3.0, 5.0, 8.0] # task1-3限定


logger.info('Prosess start!!')

# 3年分のシミュレーションを行い、真値を作成する。
l96 = Model_L96(N, F, dt, n_step)
analyzer = Analysis_Methods()
#Xn = analyzer.analyze_model(l96, MODE_SELECT)
#logger.debug(Xn)
Xn_list = analyzer.analyze_models(l96, MODE_SELECT, F_list)# task1-3限定

# グラフ出力
file_path = "./q1/result-3/"
plot = Plot_Methods()
#plot.xy_graph_l96(Xn, n_step, file_path)
plot.xy_graph_hovmoller(Xn_list, file_path)

logger.info('Prosess finish!!')
