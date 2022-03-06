# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

class PlotGlaph:
    def show_plot():
        print("a")

class RK4:
    def __init__(self):
        # T-時間, n-分割数, h-一区間の幅, t-グラフのx軸
        self.T = 50
        self.n = 1000
        self.h = self.T/self.n
        self.t = np.arange(0, self.T, self.h)
        
        # f-方程式 ( 引数1, 引数2 : 式)、引数は指定がない場合初期値をいれられる。(t=0)
        self.f = lambda u, t=0 : u 

        # 結果用の配列
        self.u = np.empty(self.n)
        self.u[0] = 1.0
    
    def calculate(self):
        for i in range (self.n-1):
            k_1 = self.h * self.f(self.u[i], self.t[i])
            k_2 = self.h * self.f(self.u[i] + k_1/2.0, self.t[i] + self.h/2.0)
            k_3 = self.h * self.f(self.u[i] + k_2/2.0, self.t[i] + self.h/2.0)
            k_4 = self.h * self.f(self.u[i] + k_3, self.t[i])
            self.u[i+1] = self.u[i] + 1/6 * (k_1 + 2.0*k_2 + 2.0*k_3 + k_4)
        return self.u

# インスタンス作成
rk4 = RK4()
plotglaph = PlotGlaph()

# 計算
data = rk4.calculate()

print (data)
# 出力
#PlotGlaph(data)