import matplotlib.pyplot as plt
import os
import numpy as np
import scipy as sp
import sympy
from sympy.codegen.cfunctions import log10
from scipy.stats import norm
from scipy.interpolate import interp1d
from logging import getLogger, DEBUG, basicConfig
logger = getLogger(__name__)
class Plot_Methods:
    def make_file(self, file_path):
        try:
            os.mkdir(file_path)
        except FileExistsError:
            pass

    def graph_noise(self, data,  file_path):
        self.make_file(file_path)
        weights = np.ones_like(data) / len(data)
        # plt.hist(data)  # 縦軸が度数
        plt.hist(data, weights = weights, density=True) # 縦軸が相対度数
        #plt.show()
        plt.savefig(file_path + "gauss_hist_0.001" + ".png")
        plt.close()
    
    def VarianceInfration(self, rmse, spread, t, day, file_path):
        fig = plt.figure()
        self.make_file(file_path)
        logger.debug('--- rmse ---')
        logger.debug(rmse.shape)
        logger.debug('--- spread---')
        logger.debug(spread.shape)
        logger.debug('--- t---')
        logger.debug(t.shape)
        logger.info('------------------------------')

        plt.xticks(np.arange(0, day, step=25))
        plt.plot(t[0:day * 4 + 1], rmse)
        plt.plot(t[0:day * 4 + 1], spread)
        plt.grid(color='k', linestyle='dotted', linewidth=0.5)
        file_name = file_path +"result-RMSE" + ".png"
        plt.savefig(file_name)
        plt.close()

    def funcOfTime(self, X, Y, X1, X2, t, day, No, file_path):
        fig = plt.figure()
        self.make_file(file_path)
        plt.xticks(np.arange(0, day + 1, step=10))
        plt.plot(t[0:day * 4 + 1], X[No, 0:day * 4 + 1], label="Truth")
        plt.plot(t[0:day * 4 + 1], Y[No, 0:day * 4 + 1], label="Observe")
        plt.plot(t[0:day * 4 + 1], X1[No, 0:day * 4 + 1], label="Forcast")
        plt.plot(t[0:day * 4 + 1], X2[No, 0:day * 4 + 1], label="Analysis")
        plt.legend()
        plt.grid(color='k', linestyle='dotted', linewidth=0.5)
        file_name = file_path +"result-funcOfTime" + ".png"
        plt.savefig(file_name)
        plt.close()