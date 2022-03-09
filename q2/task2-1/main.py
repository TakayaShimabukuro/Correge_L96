from model_l96 import Model_L96
from logging import getLogger, config
import json

''' [code sample]
logger.info('Process Start!')
logger.debug('debug')
logger.info('info')
logger.warning('warning')
logger.error('error')
logger.info('Process End!')
'''
#logの初期設定
log_config = open('./log_config.json', 'r')
config.dictConfig(json.load(log_config))
logger = getLogger(__name__)

#パラメータ設定
N = 40
F = 8.0
dt = 0.05
n_step = 5000 

l96 = Model_L96(N, F, dt, n_step)