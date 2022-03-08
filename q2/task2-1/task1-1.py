# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from copy import copy

# パラメータ
# days-日数, N-データ点の個数, dt-幅, F-外力パラメータ, n_step-ステップ数
N = 41
F = 8.0
DAYS = 2
DT = 0.05

class Model_L96:
    def __init__(self, N, F, DAYS, DT):
        
        self.N = N
        self.F = F
        self.days = DAYS
        self.dt = DT

        self.Xn = np.zeros((self.N, self.N))
        self.t = np.zeros((self.N))
        
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

    def cal_RK2(self, X_old):
        k1 = self.f_l96(X_old)
        k2 = self.f_l96(X_old + k1*self.dt)
        X_new = X_old + self.dt/2.0 * (k1 + k2)
        return X_new

    def cal_Euler(self, X_old):
        k1 = self.f_l96(X_old)
        X_new = X_old + self.dt * k1
        return X_new

    def get_estimated_data(self, MODE_SELECTED_FLG):
        # 初期値X_iのotherwiseを生成
        X = float(self.F)*np.ones(self.N)

        # i = N/2の場所だけノイズを加える
        X[20] = 1.001*self.F

        # Xnの初期値を代入
        for i in range(self.N):
            self.Xn[i, 0] = copy(X[i]) 
        
        # Xnの二回目以降を計算
        for j in range(1, self.N):
            if MODE_SELECTED_FLG == 1:
                X = self.cal_RK4(X)
            if MODE_SELECTED_FLG == 2:
                X = self.cal_RK2(X)
            if MODE_SELECTED_FLG == 3:
                X = self.cal_Euler(X)
            self.Xn[:, j] = X[:]
            self.t[j] = self.dt*j
        
        return self.Xn

def show_graph(xn, yn, zn):
    fig, ax = plt.subplots(1, 1)
    ax.set_title('L96')
    ax.set_xlabel('site')
    ax.set_ylabel('time(days)')
    ax.invert_yaxis()
    plt.xlim(0, 40)
    plt.ylim(2.3, -0.05)
    plt.grid(color='k', linestyle='dotted', linewidth=1)

    for i in range(0, 9):
        ax.plot(-5 * (xn[:, i]-F) + i*0.25)
        ax.plot(-5 * (yn[:, i]-F) + i*0.25)
        ax.plot(-5 * (zn[:, i]-F) + i*0.25)
        
    ax.legend()
    plt.show()
    fig.savefig("result_ALL_F-8.png") 
    
# インスタンス作成
l96_rk4 = Model_L96(N, F, DAYS, DT)
l96_rk2 = Model_L96(N, F, DAYS, DT)
l96_euler = Model_L96(N, F, DAYS, DT)

# 計算
xn = l96_rk4.get_estimated_data(1)
yn = l96_rk2.get_estimated_data(2)
zn = l96_euler.get_estimated_data(3)

# 出力
show_graph(xn, yn, zn)