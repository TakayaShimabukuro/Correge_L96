import matplotlib.pyplot as plt
import os
import numpy as np
import scipy as sp
import sympy
from sympy.codegen.cfunctions import log10
from scipy.stats import norm


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
    
    def graph_simulation(self, data,  file_path):
        self.make_file(file_path)
        plt.plot(data)
        plt.grid(color='k', linestyle='dotted', linewidth=0.5)
        file_name = file_path + "result" + ".png"
        plt.savefig(file_name)
        plt.close()
    
    def RMSE(self, rmse1, rmse2, t, file_path):
        fig = plt.figure()
        self.make_file(file_path)
        #plt.xticks(np.arange(0, 101, step=10))
        #plt.plot(rmse1)
        plt.plot(rmse2)
        plt.grid(color='k', linestyle='dotted', linewidth=0.5)
        file_name = file_path +"-rmse" + ".png"
        plt.savefig(file_name)
        plt.close()
    

    def funcOfTime(self, Y, X1, X2, t, file_path):
        fig = plt.figure()
        self.make_file(file_path)
        plt.xticks(np.arange(0, 101, step=10))
        #plt.plot(rmse1)
        plt.plot(t, Y[0, :])
        #plt.plot(X1)
        #plt.plot(X2)
        plt.grid(color='k', linestyle='dotted', linewidth=0.5)
        file_name = file_path +"-rmse" + ".png"
        plt.savefig(file_name)
        plt.close()
