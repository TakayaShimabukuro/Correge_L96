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
    
    def VarianceInfration(self, data, params, names):
        plt.figure()
        self.make_file(names[0])
        plt.xticks(np.arange(params[0], params[1], step=25))
        plt.plot(data[2][params[0]:params[1]*4+1], data[0][params[0]:params[1]*4+1])
        plt.plot(data[2][params[0]:params[1]*4+1], data[1][params[0]:params[1]*4+1])
        plt.grid(color='k', linestyle='dotted', linewidth=0.5)
        plt.xlabel(names[2])
        plt.ylabel(names[3])
        plt.title(names[4])
        plt.savefig(names[0] + names[1])
        plt.close()

    def funcOfTime(self, data, params, names):
        plt.figure()
        self.make_file(names[0])
        plt.xticks(np.arange(params[0], params[1], step=10))
        plt.plot(data[4][params[0]:params[1]*4+1], data[0][1, params[0]:params[1]*4+1], label="Truth")
        plt.plot(data[4][params[0]:params[1]*4+1], data[1][1, params[0]:params[1]*4+1], label="Observe")
        plt.plot(data[4][params[0]:params[1]*4+1], data[2][1, params[0]:params[1]*4+1], label="Forcast")
        plt.plot(data[4][params[0]:params[1]*4+1], data[3][1, params[0]:params[1]*4+1], label="Analysis")
        plt.legend()
        plt.grid(color='k', linestyle='dotted', linewidth=0.5)
        plt.xlabel(names[2])
        plt.ylabel(names[3])
        plt.title(names[4])
        plt.savefig(names[0] + names[1])
        plt.close()