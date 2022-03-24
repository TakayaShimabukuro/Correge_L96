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
        plt.plot(data[2][params[0]:params[1]*4+1], data[0][params[0]:params[1]*4+1], label="RMSE")
        plt.plot(data[2][params[0]:params[1]*4+1], data[1][params[0]:params[1]*4+1], label="SPREAD")
        plt.legend()
        plt.grid(color='k', linestyle='dotted', linewidth=0.5)
        plt.xlabel(names[2])
        plt.ylabel(names[3])
        plt.title(names[4])
        plt.savefig(names[0] + names[1])
        plt.close()

    def VarianceInfrationDelta(self, data, params, names):
        plt.figure()
        self.make_file(names[0])
        plt.xticks(np.arange(params[0], params[1], step=params[2]))
        for i in range(params[3]):
            label = "delt=" + str(names[5][i])
            logger.info(label)
            plt.plot(data[1][params[0]*4+1:params[1]*4+1], data[0][i, params[0]*4+1:params[1]*4+1], label=label)
        plt.grid(color='k', linestyle='dotted', linewidth=0.5)
        plt.xlim(params[0],params[1]-1)
        plt.legend()
        plt.xlabel(names[2])
        plt.ylabel(names[3])
        plt.title(names[4])
        plt.savefig(names[0] + names[1])
        plt.close()

    def VarianceInfrationDeltaPickUp(self, data, params, names):
        plt.figure()
        self.make_file(names[0])
        plt.xticks(np.arange(params[0], params[1], step=params[2]))
        for i in range(params[3]):
            label = "delt=" + str(names[5][i])
            logger.info(label)
            plt.plot(data[1][params[0]*4+1:params[1]*4+1], data[0][i, params[0]*4+1:params[1]*4+1], label=label)
        plt.grid(color='k', linestyle='dotted', linewidth=0.5)
        plt.xlim(params[0],params[1]-1)
        try:
            if params[4]:
                plt.ylim(params[5],params[6])
        except:
            pass

        plt.legend()
        plt.xlabel(names[2])
        plt.ylabel(names[3])
        plt.title(names[4])
        plt.savefig(names[0] + names[1])
        plt.close()
    

    def X1asFuncOfTime(self, data, params, names):
        plt.figure()
        self.make_file(names[0])
        plt.xticks(np.arange(params[0], params[1], step=params[2]))
        plt.plot(data[4][params[0]*4+1:params[1]*4+1], data[0][1, params[0]*4+1:params[1]*4+1], label="Truth")
        plt.plot(data[4][params[0]*4+1:params[1]*4+1], data[1][1, params[0]*4+1:params[1]*4+1], label="Observe")
        plt.plot(data[4][params[0]*4+1:params[1]*4+1], data[2][1, params[0]*4+1:params[1]*4+1], label="Forcast")
        plt.plot(data[4][params[0]*4+1:params[1]*4+1], data[3][1, params[0]*4+1:params[1]*4+1], label="Analysis")
        plt.legend()
        plt.grid(color='k', linestyle='dotted', linewidth=0.5)
        plt.xlim(params[0],params[1]-1)
        plt.xlabel(names[2])
        plt.ylabel(names[3])
        plt.title(names[4])
        plt.savefig(names[0] + names[1])
        plt.close()
    
    def aveRMSERatio(self, data, params, names):
        plt.figure()
        self.make_file(names[0])
        plt.plot(data[0], data[1], label="RMSE(analysis-true)", marker="o",fillstyle='none', color="b", linestyle="dashed", lw=0.8)
        plt.legend()
        plt.grid(color='k', linestyle='dotted', linewidth=0.5)
        plt.xlim(params[0],params[1])
        plt.xlabel(names[2])
        plt.ylabel(names[3])
        plt.title(names[4])
        plt.savefig(names[0] + names[1])
        plt.close()