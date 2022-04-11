from ctypes.wintypes import HACCEL
import numpy as np
from logging import getLogger, DEBUG, basicConfig
from plot import Plot_Methods
logger = getLogger(__name__)

class Model_L96:
    # 初期値
    def __init__(self, N, F, dt, delta, d):
        self.N = N
        self.F = F
        self.dt = dt
        self.delta = delta 
        self.d = d

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
    
    # PO法によるEnKF
    def EnKF_PO(self, Y, m):
        # init setting
        step = len(Y[0])
        m_len = len(m)
        Xb_sum = np.zeros((self.N, step))
        Xb_mean = np.zeros((self.N, step))
        H = np.identity(self.N)
        R = np.identity(self.N)
        I = np.identity(self.N)
        M = np.zeros((self.N, self.N))
        Xbs = []
        Pbs = []
        Xas = []
        Pas = []

        for i in range(m_len):
            Xbs.append(np.zeros((self.N, step)))
            Pbs.append(np.zeros(((self.N, self.N, step))))
            Xas.append(np.zeros((self.N, step)))
            Pas.append(np.zeros(((self.N, self.N, step))))


        # 初期値代入
        for i in range(m_len):
            Xbs[i][:, 0] = Y[:, m[i]]
            Pbs[i][:, :, 0] = np.diag([m[i]]*self.N)
            Xas[i][:, 0] = Y[:, m[i]]
            Pas[i][:, :, 0] = np.diag([m[i]]*self.N)
        Xb_sum = Xas[i]

        # 1. Ensemble Prediction (state)
        for i in range(m_len):
            for t in range(1, step):
                Xbs[i][:, t] = self.RK4(Xas[i][:, t-1])

            Xb_sum[:, :] += Xbs[i][:, :]
        Xb_mean = Xb_sum / m_len

        for i in range(m_len):
            Xbs[i] -= Xb_mean
        
        Zb = Xbs / np.sqrt(m_len-1)

        # 2. Prediction of Error Covariance (implicitly)
        # 3. Kalman Gain
        # 4. Analysis (state)
        # 5. Analysis Error Covariance