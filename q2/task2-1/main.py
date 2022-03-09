from model_l96 import Model_L96
from logging import getLogger, config, DEBUG, basicConfig
import json

''' [code sample]
logger.info('Process Start!')
logger.debug('debug')
logger.info('info')
logger.warning('warning')
logger.error('error')
logger.info('Process End!')
'''
# logの初期設定
log_config = open('./log_config.json', 'r')
config.dictConfig(json.load(log_config))
logger = getLogger(__name__)
logger.setLevel(DEBUG)
basicConfig(filename='console.log', level=DEBUG)



# パラメータ設定
N = 40
F = 8.0
dt = 0.05
n_step = 5000
MODE_SELECT = 3  # 1-Euler, 2-RK2, 3, RK4

# 3年分のシミュレーションを行い、真値を作成する。
l96 = Model_L96(N, F, dt, n_step)
l96.get_estimated_data(MODE_SELECT)

logger.info('Prosess finish')
