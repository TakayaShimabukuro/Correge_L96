# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

class PlotGlaph:
    def show_plot():
        print("a")

class RK4:
    def init():
        # t-時間, n-分割数, h-一区間の幅, グラフのx軸, f-方程式
        t = 50  
        n = 1000
        h = t/n
        x_axis = np.arange(0, t, h)
        f = 1
    
    def calculate():
        return 0

# インスタンス作成
rk4 = RK4()
plotglaph = PlotGlaph()

# 計算
rk4.init()
data = rk4.calculate()

# 出力
PlotGlaph(data)