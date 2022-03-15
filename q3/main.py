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

logger = getLogger(__name__)                                                            # logの初期設定
logger.setLevel(DEBUG)
basicConfig(filename='console.log', level=DEBUG, filemode='w')


N = 41                                                                                  # public parameter
F = 8.0
dt = 0.05
std_list = [0.001]


logger.info('Prosess Start!!')                                                          # local parameter
step_2year = 2848                                                                       # 4*356*2
step_1year = 1424                                                                       # 4*356
n_step_true = 4000                                                                      # process 2
n_step_forcast_init = 1000                                                              # process 3, 4
n_step_forcast = 201                                                                    # process 3, 4, 5, 400step - 100day

for k in range(len(std_list)):


    logger.info('Prosess 1')                                                            # 1. L96を2年分シミュレーションする
    l96 = Model_L96(N, F, dt, std_list[k])
    Xn_2year = np.zeros((N, step_2year))
    X1_2year = float(F)*np.ones(N)
    X1_2year[20] = 1.001*F
    Xn_2year, tn_2year = l96.analyze_model(Xn_2year, X1_2year, step_2year)

    
    logger.info('Prosess 2')                                                            # 2. 後半1年分を6時間毎に保存する
    Xn_1year = np.zeros((N, step_1year))
    Xn_1year = Xn_2year[:, step_1year:step_2year]
   

    logger.info('Prosess 3')                                                            # 3. 真値の1-1000stepをそれぞれ初期値とし,　そこにノイズを加え、時間経過
   


logger.info('Prosess finish')
