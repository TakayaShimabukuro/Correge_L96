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
        self.Xn_noise = []
        self.Xn_pred = np.zeros((self.DAYS2, self.N, self.DAYS2))
        self.t = np.zeros(self.DAYS)
        self.t_true = np.zeros(self.DAYS)
        self.t_pred = np.zeros((self.DAYS2, self.DAYS))

        self.std = 0.01

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

        for i in range(self.N):
            self.Xn[i, 0] = copy(X[i])

        for j in range(1, int(self.DAYS)):
            X = self.cal_RK4(X)
            self.Xn[:, j] = X[:]
            self.t[j] = self.dt*j
    
    def get_estimated_data_true(self):
        X = self.Xn[:, int(self.DAYS)-1]
        for i in range(self.N):
            self.Xn_true[i, 0] = copy(X[i])

        for j in range(1, int(self.STEP)):
            X = self.cal_RK4(X)
            self.Xn_true[:, j] = X[:]
            self.t_true[j] = self.dt*j

    def get_estimated_data_pred(self):
        for i in range(self.DAYS2):
            X = self.Xn_noise[i]
            for j in range(self.N):
                self.Xn_pred[i][j, 0] = copy(X[j])

            for k in range(1, int(self.DAYS2)):
                X = self.cal_RK4(X)
                self.Xn_pred[i][:, k] = X[:]
                self.t_pred[i][k] = self.dt*k
    
    def add_noise(self):
        noise = np.random.normal(loc=0, scale=1, size=40)
        tmp = []
        for i in range(0, int(self.DAYS2)):
            tmp = self.Xn_true[:, i] + noise
            self.Xn_noise.append(tmp)
    
    def get_rsme(self):
        tmp = []
        for i in range(self.DAYS2):
            tmp.append(self.Xn_true[:, i:i+self.DAYS2] - self.Xn_pred[i])
        
        tmp = np.array(tmp)
        rmse = np.mean(np.sqrt(np.mean(np.power(tmp, 2), axis = 2)), axis = 0)

        return rmse

def show_graph(rsme):
    fig = plt.figure()
    plt.plot(rsme)
    plt.xlabel("time(days)")
    plt.ylabel("RSME")
    plt.legend()
    plt.xticks(np.arange(0, 110, step=10))
    plt.grid(color='k', linestyle='dotted', linewidth=0.5)
    plt.savefig("result.png")
    plt.close()



l96_rk4 = Model_L96()

# 1.3年分のシミュレーション
l96_rk4.get_estimated_data() 

# 2.3年分からスタートして300ステップの真値を取得
l96_rk4.get_estimated_data_true()

# 3.真値の各ステップにノイズを付加。
l96_rk4.add_noise()

# 4.ノイズ付加された配列を100STEP時間を進める。
l96_rk4.get_estimated_data_pred()

# RSME 計算
rmse = l96_rk4.get_rsme()

# 出力
show_graph(rmse)
