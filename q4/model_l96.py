from ctypes.wintypes import HACCEL
import numpy as np
from logging import getLogger, DEBUG, basicConfig
logger = getLogger(__name__)

class Model_L96:
    def __init__(self, N, F, dt, delta):
        self.N = N
        self.F = F
        self.dt = dt
        self.delta = delta 

        
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
    def KF(self, Y, d):
        
        # init setting
        step = len(Y[0])
        Xf = np.zeros((self.N, step))
        Pf = np.zeros(((self.N, self.N, step)))
        Xa = np.zeros((self.N, step))
        Pa = np.zeros(((self.N, self.N, step)))
        H = np.identity(self.N)
        R = np.identity(self.N)
        I = np.identity(self.N)
        M = np.zeros((self.N, self.N))

        # progress 1
        Xf[:, 0] = Y[:, 100]
        Pf[:, :, 0] = np.diag([10]*self.N)
        Xa[:, 0] = Y[:, 100]
        Pa[:, :, 0] = np.diag([10]*self.N)
        
        for t in range(1, step):
            # progress 2
            M = self.get_M(Xa[:, t-1])            
            Xf[:, t] = self.RK4(Xa[:, t-1])
            Pf[:, :, t] = (M@Pa[:, :, t-1]@M.T)*(1 + d)

            # progress 3
            K = (Pf[:, :, t]@H.T)@np.linalg.inv(H@Pf[:, :, t]@H.T + R)
            Xa[:, t] = Xf[:, t] + K@(Y[:,t-1] - Xf[:, t])
            Pa[:, :, t] = (I-K@H)@Pf[:, :, t]

        # progress 4
        return Xf, Pf, Xa, Pa


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
    
    def showAveRMSE(self, data, d):
        logger.info("Ave RMSE (10th day - 300th day)")
        logger.info("-------------------------------")
        for i in range(len(d)):
            ave = sum(data[i]) / len(data[i])
            logger.debug("delta=" + str(d[i]) + "-> RMSE=" + str(ave))
    
    def RMSE(self, X1, X2, step):
        rmse = np.zeros((step))
        for i in range(step):
            sub = X1[:, i] - X2[:, i]
            rmse[i] = np.sqrt(np.mean(sub**2))
        return rmse

    def Spread(self, P):
        return np.sqrt(np.trace(P)/len(P))

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
    