from ctypes.wintypes import HACCEL
import numpy as np
from logging import getLogger, DEBUG, basicConfig
logger = getLogger(__name__)
class Model_L96:
    def __init__(self, N, F, dt):
        self.N = N
        self.F = F
        self.dt = dt

        
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
        return X1 + (k1 + 2.0*k2 + 2.0*k3 + k4) /6.0

    # カルマンフィルタ
    def KF(self, Y):
        
        # init setting
        step = len(Y)
        delta = 10**-5
        Xf = np.zeros((self.N, step))
        Pf = np.zeros((self.N, self.N, step))
        Xa = np.zeros((self.N, step))
        Pa = np.zeros((self.N, self.N, step))
        H = np.identity(self.N)
        R = np.identity(self.N)
        I = np.identity(self.N)
        M = np.zeros((self.N, self.N))

        # progress 1
        Xf[:, 0] = Y[:, 100]
        Pf[:, :, 0] = np.diag([25]*self.N)
        Xa[:, 0] = Y[:, 100]
        Pa[:, :, 0] = np.diag([25]*self.N)

        for t in range(1, step):
            # progress 2
            M = self.get_M(Xa[:, t-1], delta)
            Xf[:, t] = self.RK4(Xa[:, t-1])
            Pf[:, :, t] = M@Pa[:, :, t-1]@M.T

            # progress 3
            a = Pf[:, :, t]@H.T
            b = H@Pf[:, :, t]@H.T + R
            K = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
            Xa[:, t] = Xf[:, t] + K@(Y[:,t-1] - Xf[:, t])
            Pa[:, :, t] = (I - K@H)@Pf[:, :, t]@(I - K@H).T+K@R@K.T

        # progress 4
        return Xf, Pf, Xa, Pa


    # Mを導出
    def get_M(self, X, delta):
        e = np.identity(self.N)
        M = np.zeros((self.N, self.N))

        for i in range(self.N):
            M[:, i] = (self.RK4(X + delta*e[:, i])-self.RK4(X)) / delta
            
        return M
    
    
    def RMSE(self, X1, X2):
        rmse= [] # 1000シミュレーションを平均したrmseを格納、100-step分出力される
        for i in range(len(X1)):
            sub = X1[:, i] - X2[:, i]
            rmse.append(np.sqrt(np.mean(sub**2)))
        return rmse


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
    