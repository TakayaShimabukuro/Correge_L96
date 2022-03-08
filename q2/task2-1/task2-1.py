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
DAYS = 3000

class RndnumBoxMuller:
    M = 1        # 平均
    S = 2.5       # 標準偏差
    N = 10000     # 生成する個数
    SCALE = N // 100  # ヒストグラムのスケール
    std = 0.001
    n_round = 3


    def generate_rndnum(self):
        # ガウス分布に従う乱数をBox muller法で作成。式は以下。
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
            
            x = round(x*self.std, self.n_round)
            y = round(y*self.std, self.n_round)
            #print([x, y])
            return [x, y]
        except Exception as e:
            raise


class Model_L96:
    def __init__(self, N, DAYS, F, DT):
        
        self.N = N
        self.F = F
        self.dt = DT
        self.DAYS = DAYS

        self.Xn = np.zeros((self.N, self.DAYS))
        self.t = np.zeros((self.DAYS))
    
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
        for j in range(1, self.DAYS):
            X = self.cal_RK4(X)
            self.Xn[:, j] = X[:]
            self.t[j] = self.dt*j
        
        return self.Xn

def show_graph(xn):
    for i in range(0, DAYS):
        print(xn[:, i])
        if i == DAYS-1:
            fig = plt.figure()
            plt.plot(xn[:, i])
            plt.xlim(0, 40)
            plt.show()


# インスタンス作成

l96_rk4 = Model_L96(N, DAYS, F, DT)
l96_rk2 = Model_L96(N, DAYS, F, DT)
l96_euler = Model_L96(N, DAYS, F, DT)

# 計算
xn = l96_rk4.get_estimated_data()

# 出力
show_graph(xn)