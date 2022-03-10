# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
import os
import math
import random
from sklearn.metrics import mean_squared_error


class Model_L96:
    def __init__(self):

        self.N = 40
        self.F = 8.0
        self.dt = 0.05
        self.DAYS = 5000
        self.DAYS2 = 100
        self.STEP = 300

        self.Xn = np.zeros((self.N, self.DAYS))
        self.Xn_true = np.zeros((self.N, self.STEP))
        self.Xn_pred = np.zeros((self.N, self.DAYS2))
        self.Xn_noise = []

    def f_l96(self, x):
        f = np.zeros((self.N))
        for i in range(2, self.N-1):
            f[i] = (x[i+1]-x[i-2])*x[i-1]-x[i]+self.F

        f[0] = (x[1]-x[self.N-2])*x[self.N-1]-x[0]+self.F
        f[1] = (x[2]-x[self.N-1])*x[0]-x[1]+self.F
        f[self.N-1] = (x[0]-x[self.N-3])*x[self.N-2]-x[self.N-1]+self.F

        return f

    def cal_RK4(self, X_old):
        k1 = self.f_l96(X_old)
        k2 = self.f_l96(X_old + k1*self.dt/2.0)
        k3 = self.f_l96(X_old + k2*self.dt/2.0)
        k4 = self.f_l96(X_old + k3*self.dt)
        X_new = X_old + self.dt/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)
        return X_new

    def get_estimated_data(self):
        X = float(self.F)*np.ones(self.N)
        X[20] = 1.001*self.F
        self.Xn = self.create_table(self.N, self.DAYS, X, self.Xn)

    def get_estimated_data_true(self):
        X = self.Xn[:, self.DAYS-1]
        self.Xn_true = self.create_table(self.N, self.STEP, X, self.Xn_true)
        #self.show_graph(self.Xn_true, "Xn_true")

    def get_estimated_data_pred(self):
        tmp = []
        for i in range(self.DAYS2):
            X = self.Xn_noise[i]
            self.Xn_pred = self.create_table_noise(self.N, self.DAYS2, X, self.Xn_pred)
            #self.show_graph(self.Xn_pred, i)
            tmp.append(self.Xn_pred)
        return tmp

    def add_noise(self):
        noise = np.random.normal(loc=0, scale=1, size=40)
        tmp = []
        for i in range(0, self.DAYS2):
            tmp = self.Xn_true[:, i] + noise
            self.Xn_noise.append(tmp)

    def get_rmse(self, data_pred):
        rmse_tmp = []
        for i in range(self.DAYS2):
            for j in range(self.DAYS2):
                pass
                #self.show_graph(data_pred[i], i)
                #self.show_graph(self.Xn_true[:, i:i+self.DAYS2], i)
                #tmp = np.sqrt(mean_squared_error(self.Xn_true[:, i:i+self.DAYS2], data_pred[i][:, :]))
                #rmse_tmp.append(tmp)
        

        return rmse_tmp

    def create_table(self, n, step, X, tmp_X):
        for i in range(n):
            tmp_X[i, 0] = copy(X[i])
        for j in range(1, step):
            X = self.cal_RK4(X)
            tmp_X[:, j] = X[:]
        return tmp_X
    
    def create_table_noise(self, n, step, X, tmp_X):
        noise = np.random.normal(loc=0, scale=1, size=40)
        for i in range(n):
            tmp_X[i, 0] = copy(X[i])
        for j in range(1, step):
            X += noise
            X = self.cal_RK4(X)
            tmp_X[:, j] = X[:]
        
            #self.show_graph(tmp_X, j)
        return tmp_X

    def show_graph(self, rmse, i):
        fig = plt.figure()
        try:
            new_dir_path = './q2/' + str(i)
            os.mkdir(new_dir_path)
        except FileExistsError:
            pass
        for j in range(0, self.DAYS2):
            if j%10 == 0:
                plt.plot(rmse[:, j])
                plt.xlabel("time(days)")
                plt.ylabel("RMSE")
                #plt.xticks(np.arange(0, 110, step=10))
                plt.grid(color='k', linestyle='dotted', linewidth=0.5)
                name =  "./q2/" + str(i) + "/" + str(j) + ".png"
                plt.savefig(name)
                plt.close()

def show_graph(rmse):
    fig = plt.figure()
    plt.plot(rmse)
    plt.xlabel("time(days)")
    plt.ylabel("RMSE")
    plt.legend()
    #plt.xticks(np.arange(0, 110, step=10))
    plt.grid(color='k', linestyle='dotted', linewidth=0.5)
    plt.savefig("./q2/result.png")
    plt.close()


l96_rk4 = Model_L96()

# 1.3年分のシミュレーション 40 (element) * 5000 (time)
l96_rk4.get_estimated_data()

# 2.3年分からスタートして300ステップの真値を取得 40 (element) * 300 (time)
l96_rk4.get_estimated_data_true()
# print(len(l96_rk4.Xn_true[0]))
# print(l96_rk4.Xn_true)

# 3.真値の各ステップにノイズを付加。 100 (time) * 40 (element) * 1 (time)
l96_rk4.add_noise()

# 4.ノイズ付加された配列を100STEP時間を進める。
data_pred = l96_rk4.get_estimated_data_pred()
#print(len(data_pred[0]))
#print(data_pred)

# RMSE 計算
rmse = l96_rk4.get_rmse(data_pred)
print(len(rmse))
# 出力
#show_graph(rmse)
