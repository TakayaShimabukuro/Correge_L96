import numpy as np
class Model_L96:
    def __init__(self, N, F, dt, std):
        self.N = N
        self.F = F
        self.dt = dt
        self.std = std
        
    # Lorenz 96
    def l96(self, x):
        f = np.zeros_like(x)
        f[0] = (x[1]-x[self.N-2])*x[self.N-1]-x[0]+self.F
        f[self.N-1] = (x[0]-x[self.N-3])*x[self.N-2]-x[self.N-1]+self.F
        for i in range(1, self.N-1):
            f[i] = (x[i+1]-x[i-2])*x[i-1]-x[i]+self.F
        return f
    
    # ルンゲクッタ4次
    def RK4(self, X1):
        k1 = self.l96(X1) * self.dt
        k2 = self.l96(X1 + k1*0.5) * self.dt
        k3 = self.l96(X1 + k2*0.5) * self.dt
        k4 = self.l96(X1 + k3) * self.dt
        return X1 + (k1 + 2*k2 + 2*k3 + k4) / 6.0

    # RMSEを計算するメソッド
    def get_RMSE(self, Xn_true, Xn_forcast, step):
        rmse = 0 # 1stepの1000シミュレーションのrmseを格納
        rmse_step = [] # 1000シミュレーションを平均したrmseを格納、100-step分出力される
        for i in range(step):
            for j in range (len(Xn_true)):
                sub = Xn_true[j][:, i] - Xn_forcast[j][:, i]
                rmse += np.sqrt(np.mean(sub**2))
            rmse_step.append(rmse/len(Xn_true))

        return rmse_step
    
    def get_RMSE_solo(self, Xn_true, Xn_forcast, step, num):
        rmse= [] # 1000シミュレーションを平均したrmseを格納、100-step分出力される
        for i in range(step):
            sub = Xn_true[num][:, i] - Xn_forcast[num][:, i]
            rmse.append(np.sqrt(np.mean(sub**2)))
        return rmse

    # 配列にノイズを付加するメソッド
    def add_noise(self, X1):
        noise = np.random.normal(loc=0, scale=1, size=len(X1))
        X1_tmp = np.zeros(self.N)
        X1_tmp[:] = X1[:] + (noise*self.std)
        return X1_tmp

    # X1[40, 1]を初期値とするstep分のシミュレーションを行う。
    def analyze_model(self, Xn, X1_tmp, step):
        t = np.zeros(step)
        X1 = np.zeros(self.N)
        X1[:] = X1_tmp[:]
        for i in range(self.N):
            Xn[i, 0] = X1[i]
        for j in range(1, step):
            X1 = self.RK4(X1)
            Xn[:, j] = X1[:]
            t[j] =self.dt*j*5
        return Xn, t
    