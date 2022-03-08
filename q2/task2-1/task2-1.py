# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
import os
import math
import random
from sklearn.metrics import mean_squared_error
# パラメータ
# dN-データ点の個数, dt-幅, F-外力パラメータ, n_step-ステップ数, 2000, 8000, 20000

F = 8.0
DT = 0.05
N = 40
DAYS = 1000  # 21900

class RndnumBoxMuller:
    M = 20      # 平均
    S = 2.5    # 標準偏差
    N = 40     # 生成する個数
    std = 0.001
    n_round = 3

    def __init__(self, X):
        self.hist = X

    def generate_rndnum(self):
        try:
            for _ in range(self.N):
                res = self.__rnd()
                self.hist[res[0]] += self.std
                self.hist[res[1]] += self.std

            return self.hist
        except Exception as e:
            raise

    def __rnd(self):
        try:
            r_1 = random.random()
            r_2 = random.random()
            x = self.S \
                * math.sqrt(-2 * math.log(r_1)) \
                * math.cos(2 * math.pi * r_2) \
                + self.M
            y = self.S \
                * math.sqrt(-2 * math.log(r_1)) \
                * math.sin(2 * math.pi * r_2) \
                + self.M
            return [math.floor(x), math.floor(y)]
        except Exception as e:
            raise


class Model_L96:
    def __init__(self, N, DAYS, F, DT):

        self.N = N
        self.F = F
        self.dt = DT
        self.DAYS = DAYS

        self.Xn = np.zeros((self.N, int(self.DAYS)))
        self.t = np.zeros(int(self.DAYS))

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
        # 初期値X_iのotherwiseを生成
        X = float(self.F)*np.ones(self.N)

        # i = N/2の場所だけノイズを加える
        X[20] = 1.001*self.F

        # Xnの初期値を代入
        for i in range(self.N):
            self.Xn[i, 0] = copy(X[i])

        # Xnの二回目以降を計算
        for j in range(1, int(self.DAYS)):
            X = self.cal_RK4(X)
            self.Xn[:, j] = X[:]
            self.t[j] = self.dt*j

        return self.Xn

    def get_estimated_data_added_noise(self, xn_true):        
        for i in range(self.N):
            box_muller = RndnumBoxMuller(xn_true)
            X = box_muller.generate_rndnum()
            self.Xn[i, 0] = copy(X[i])

        # Xnの二回目以降を計算
        print(self.DAYS)
        for j in range(1, int(self.DAYS)):
            X = self.cal_RK4(X)
            self.Xn[:, j] = X[:]
            self.t[j] = self.dt*j

def show_graph(xn):
    for i in range(0, int(DAYS)):
        if i == int(DAYS)-1:
            fig = plt.figure()
            plt.plot(xn[:, i])
            plt.xlim(0, 40)
            plt.show()


# L96 - 真値導出
l96_rk4 = Model_L96(N, DAYS, F, DT)
l96_rk4.get_estimated_data()
xn_true = l96_rk4.Xn[:, int(DAYS)-1]
#print(len(l96_rk4.Xn[0]))

# L96 - 観測値導出
l96_rk4_true = Model_L96(N, DAYS, F, DT)
xn_pred = l96_rk4_true.get_estimated_data_added_noise(xn_true)
#print(len(l96_rk4_true.Xn[0]))

# 出力
show_graph(l96_rk4_true.Xn)
