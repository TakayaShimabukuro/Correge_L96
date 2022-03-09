# 外部ライブラリ
from logging import getLogger, config, DEBUG, basicConfig
import sys, os
sys.path.append(os.path.join('..', 'code'))

# 内部ライブラリ
from model_l96 import Model_L96
from numerical_analysis import Analysis_Methods
from plot import Plot_Methods

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
F = 1.0
dt = 0.05
n_step = 1000
MODE_SELECT = 3 # 1-Euler, 2-RK2, 3, RK4

# 3年分のシミュレーションを行い、真値を作成する。
l96 = Model_L96(N, F, dt, n_step)
analyzer = Analysis_Methods()
analyzer.analyze_model(l96, MODE_SELECT)

# グラフ出力
file_path = "./q2/result/"
plot = Plot_Methods()
plot.xy_graph_l96(l96, file_path)

logger.info('Prosess finish')
