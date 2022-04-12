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
        H = np.identity(self.N)
        R = np.identity(self.N)
        I = np.identity(self.N)
        M = np.zeros((self.N, self.N))

        # m個格納用
        Xbs = []
        Xas = []
        Pbs = []
        Pas = []
        dXbs = []
        Zbs = []

        # 1次元として用いるが、転置ができなくなるため2次元で宣言
        Xb_sum = np.zeros((self.N, self.N))
        Xb_mean = np.zeros((self.N, self.N))
        
        # 1-0 準備 OK
        for i in range(m_len):
            Xbs.append(np.zeros((self.N, step)))
            Xas.append(np.zeros((self.N, step)))
            Pbs.append(np.zeros(((self.N, self.N, step))))
            Pas.append(np.zeros(((self.N, self.N, step))))

        # 1-1 Analysis Ensemble OK
        for i in range(m_len):
            Xbs[i][:, 0] = Y[:, m[i]]
            Pbs[i][:, :, 0] = np.diag([m[i]]*self.N)
            Xas[i][:, 0] = Y[:, m[i]]
            Pas[i][:, :, 0] = np.diag([m[i]]*self.N)
        
        # 1-2 Ensemble Forecasts OK
        for t in range(1, step):
            for i in range(m_len):
                Xbs[i][:, t] = self.RK4(Xas[i][:, t-1])
                Xb_sum[:, 0] = Xb_sum[:, 0] + Xbs[i][:, t]
                
            # 1-3 Ensemble Mean OK
            Xb_mean[:, 0] = Xb_sum[:, 0] / m_len
        
            # 1-4 Ensemble Perturbation, m個のdelta Xbを保持 OK
            for i in range(m_len):
                tmp_delta = np.zeros((self.N, self.N))
                tmp_delta[:, 0] = Xbs[i][:, t] - Xb_mean[:, 0]
                dXbs.append(tmp_delta)
            

            # 2 Prediction of Error Covariance (implicitly) 
            for i in range(m_len):
                tmp_Zb = np.zeros((self.N, self.N))
                tmp_Zb = dXbs[i]/np.sqrt(m_len-1)
                Pbs[i][:, :, t] = tmp_Zb@tmp_Zb.T

            
            logger.debug('Zbs[0][0, 0]={}'.format(Zbs[0][0:3, 0:3]))
            logger.debug('Zbs[1][0, 0]={}'.format(Zbs[1][0:3, 0:3]))
            self.plot.Debug(Zbs[0], "Zbs[0]")
            self.plot.Debug(Zbs[1], "Zbs[1]")


        # 2. Prediction of Error Covariance (implicitly)
        # 3. Kalman Gain
        # 4. Analysis (state)
        # 5. Analysis Error Covariance