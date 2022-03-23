import matplotlib.pyplot as plt
import os
import numpy as np
import scipy as sp
import sympy
from sympy.codegen.cfunctions import log10
from scipy.stats import norm
from scipy.interpolate import interp1d
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
    

    def funcOfTime(self, X, Y, X1, X2, t, file_path):
        fig = plt.figure()
        self.make_file(file_path)
        day = 50
        
        No = 10
        plt.xticks(np.arange(0, day + 1, step=10))

        x_1, y_1 = self.moving_avg(t[0:day * 4 + 1], X[No, 0:day * 4 + 1])
        x_2, y_2 = self.moving_avg(t[0:day * 4 + 1], Y[No, 0:day * 4 + 1])
        x_3, y_3 = self.moving_avg(t[0:day * 4 + 1], X1[No, 0:day * 4 + 1])
        x_4, y_4 = self.moving_avg(t[0:day * 4 + 1], X2[No, 0:day * 4 + 1])

        xs_1, ys_1 = self.spline_interp(x_1, y_1)
        xs_2, ys_2 = self.spline_interp(x_2, y_2)
        xs_3, ys_3 = self.spline_interp(x_3, y_3)
        xs_4, ys_4 = self.spline_interp(x_4, y_4)
        plt.plot(t[0:day * 4 + 1], X[No, 0:day * 4 + 1], label="Truth")
        plt.plot(t[0:day * 4 + 1], Y[No, 0:day * 4 + 1], label="Observe")
        plt.plot(t[0:day * 4 + 1], X1[No, 0:day * 4 + 1], label="Forcast")
        plt.plot(t[0:day * 4 + 1], X2[No, 0:day * 4 + 1], label="Analysis")
        plt.legend()
        plt.grid(color='k', linestyle='dotted', linewidth=0.5)
        file_name = file_path +"result-funcOfTime" + ".png"
        plt.savefig(file_name)
        plt.close()

    
    def spline_interp(self, in_x, in_y):
        out_x = np.linspace(np.min(in_x), np.max(in_x), np.size(in_x)*100) # もとのxの個数より多いxを用意
        func_spline = interp1d(in_x, in_y, kind='cubic') # cubicは3次のスプライン曲線
        out_y = func_spline(out_x) # func_splineはscipyオリジナルの型

        return out_x, out_y

    def moving_avg(self, in_x, in_y):
        np_y_conv = np.convolve(in_y, np.ones(3)/float(3), mode='valid') # 畳み込む
        out_x_dat = np.linspace(np.min(in_x), np.max(in_x), np.size(np_y_conv))

        return out_x_dat, np_y_conv