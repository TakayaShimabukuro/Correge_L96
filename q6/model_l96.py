from ctypes.wintypes import HACCEL
import numpy as np
from logging import getLogger, DEBUG, basicConfig
from plot import Plot_Methods
logger = getLogger(__name__)


class Model_L96:
    # 初期値
    def __init__(self, N, F, dt, delta, d, plot):
        self.N = N
        self.F = F
        self.dt = dt
        self.delta = delta
        self.d = d
        self.plot = plot

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
            t[j] = self.dt*j*5
        return Xn, t

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
        return X1 + (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0

    # Mを導出
    def get_M(self, X):
        e = np.identity(self.N)
        M = np.zeros((self.N, self.N))

        np.set_printoptions(threshold=np.inf)
        for i in range(self.N):
            a = X + self.delta*e[:, i]
            b = X
            M[:, i] = (self.RK4(a)-self.RK4(b)) / self.delta

        return M

    # RMSEを取得
    def RMSE(self, X1, X2, step):
        rmse = np.zeros((step))
        for t in range(step):
            sub = X1[:, t] - X2[:, t]
            rmse[t] = np.sqrt(np.mean(sub**2))
        return rmse

    # Spreadを取得
    def Spread(self, P, step):
        tmp = []
        for t in range(step):
            tmp.append(np.sqrt(np.trace(P[:, :, t])/len(P[:, :, t])))
        return tmp

    # Spreadを取得
    def Reshape(self, Y, step):
        YO = np.zeros(((self.N, self.N, step)))
        for t in range(step):
            YO[:, 0, t] = Y[:, t]
        return YO

    # PO法によるEnKF
    def EnKF_PO(self, Y, m_temp, noise, step):
        m_len = len(m_temp)
        Xb = np.zeros(((self.N, step, m_len)))
        Xa = np.zeros(((self.N, step, m_len)))
        Xa_mean = np.zeros((self.N, step))
        Pa = np.zeros(((self.N, self.N, step)))
        H = np.identity(self.N)
        R = np.identity(self.N)
        I = np.identity(m_len)

        # t = 0
        for m in range(m_len):
            Xb[:, 0, m] = Y[:, 100 + (m*2)]+noise
            Xa[:, 0, m] = Y[:, 100 + (m*2)]+noise
        
        # t > 0
        for t in range(1, step):
            Xb_sum = np.zeros(self.N)
            Xa_sum = np.zeros(self.N)
            Pa_sum = np.zeros((self.N, self.N))  
            dXb = np.zeros((self.N, m_len))
            dXa = np.zeros((self.N, m_len))

            for m in range(m_len):
                Xb[:, t, m] = self.RK4(Xa[:, t-1, m])
                Xb_sum = Xb_sum + Xb[:, t, m]
            Xb_mean = Xb_sum / m_len
            
            for m in range(m_len):
                dXb[:, m] = (Xb[:, t, m]-Xb_mean)
            
            Zb = dXb / np.sqrt(m_len-1)
            Zb = Zb * 1.2
            Yb = H@Zb
            K = Zb @ (np.linalg.inv(I + Yb.T@np.linalg.inv(R)@ Yb)) @ Yb.T @ np.linalg.inv(R)
            
            for m in range(m_len):                
                Xa[:, t, m] = Xb[:, t, m] + K@(Y[:, t]+noise-H@Xb[:, t, m])
                Xa_sum = Xa_sum + Xa[:, t, m]
            Xa_mean[:, t] = Xa_sum / m_len

            for m in range(m_len):
                dXa[:, m] = Xa[:, t, m] - Xa_mean[:, t]

            for m in range(m_len):
                Za = dXa/np.sqrt(m_len-1)
                Pa_sum = Pa_sum + Za@Za.T

            Pa[:, :, t] = Pa_sum / m_len

        return Xa, Xa_mean, Pa
