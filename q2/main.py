# 外部ライブラリ
from logging import getLogger, DEBUG, basicConfig
import numpy as np

# 内部ライブラリ
from plot import Plot_Methods
from model_l96 import Model_L96

# logの初期設定
logger = getLogger(__name__)
logger.setLevel(DEBUG)
basicConfig(filename='console.log', level=DEBUG, filemode='w')

# parameter
N = 41
F = 8.0
dt = 0.05
std_list = [0.001]
logger.info('Prosess Start!!')
n_step_init = 5000
n_step_true = 4000
n_step_forcast_init = 1000
n_step_forcast = 201

for k in range(len(std_list)):

    # 1. 3年分のシミュレーションを行い、真値の初期値を作成する。
    logger.info('Prosess 1')
    l96 = Model_L96(N, F, dt, std_list[k])
    Xn_init = np.zeros((N, n_step_init))
    X1_init = float(F)*np.ones(N)
    X1_init[20] = 1.001*F
    Xn_init, tn_init = l96.analyze_model(Xn_init, X1_init, n_step_init)

    # 2. 初期値Xnをもとに真値Xn[40, :]を取得
    logger.info('Prosess 2')
    Xn_true = np.zeros((N, n_step_true))
    X1_true = Xn_init[:, n_step_init-1]
    Xn_true, tn_true = l96.analyze_model(Xn_true, X1_true, n_step_true)

    # 3. 真値の1-1000stepをそれぞれ初期値とし,　そこにノイズを加え、時間経過
    logger.info('Prosess 3')
    Xn_forcasts = []
    for i in range(n_step_forcast_init):
        X1_forcast = l96.add_noise(Xn_true[:, i])
        Xn_forcast = np.zeros((N, n_step_forcast))
        Xn_forcast, tn_forcast = l96.analyze_model(Xn_forcast, X1_forcast, n_step_forcast)
        Xn_forcasts.append(Xn_forcast)

    # 4. Xn_truesをスライス
    logger.info('Prosess 4')
    Xn_trues = []
    for i in range(n_step_forcast_init):
        Xn_trues.append(Xn_true[:, i:i+n_step_forcast])

    # 5. rmseを導出する。
    logger.info('Prosess 5')
    num_list = np.arange(0, n_step_forcast_init, step=1)
    rmse_sum = [0] * n_step_forcast_init
    for i in range(n_step_forcast_init):
        rmse = l96.get_RMSE_solo(Xn_trues, Xn_forcasts,n_step_forcast, num_list[i])
        rmse_sum = list(map(lambda val1, val2: val1 + val2, rmse_sum, rmse))
        rmse_sum = list(map(lambda val: val / n_step_forcast_init,rmse_sum))                     # nで割って平均を求める

        # 6. グラフ出力
        logger.info('Prosess 6')
        file_path = "./q2/result/"
        plot = Plot_Methods()
        # plot.xy_graph_rmse(rmse_sum, file_path, tn_forcast, std_list[k])　                 # rmseのみの出力
        # log scaleも出力
        plot.xy_graph_rmse_log(rmse_sum, file_path, tn_forcast, std_list[k])

logger.info('Prosess finish')
