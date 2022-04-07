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
        Pf[:, :, 0] = np.diag([25]*self.N)
        Xa[:, 0] = Y[:, 100]
        Pa[:, :, 0] = np.diag([25]*self.N)
        
        for t in range(1, step):
            # progress 2
            M = self.get_M(Xa[:, t-1])            
            Xf[:, t] = self.RK4(Xa[:, t-1])
            Pf[:, :, t] = (M@Pa[:, :, t-1]@M.T)*(1 + d)

            # progress 3
            K = (Pf[:, :, t]@H.T)@np.linalg.inv(H@Pf[:, :, t]@H.T + R)
            Xa[:, t] = Xf[:, t] + K@(Y[:,t] - Xf[:, t])
            Pa[:, :, t] = (I-K@H)@Pf[:, :, t]

        # progress 4
        return Xf, Pf, Xa, Pa
    
    # 3DVAR
    def analyze_3DVAR(self, Y, B, step):
        Xb = np.zeros((self.N, step))
        Xa = np.zeros((self.N, step))
        H = np.identity(self.N)
        R = np.identity(self.N)
        Xa[:, 0] = Y[:, 100]

        for t in range(1, step):
            Xb[:, t] = self.RK4(Xa[:, t-1])
            K = (B@H.T)@np.linalg.inv(H@B@H.T + R)
            Xa[:, t] = Xb[:, t] + K@(Y[:,t] - H@Xb[:, t])

        return Xa
    
    # 3DVAR-case
    def analyze_3DVAR_case(self, Y_all, B, step, delate_queue):
        n_size = len(delate_queue)
        Xb = np.zeros((self.N, step))
        Xa = np.zeros((self.N, step))
        R = np.identity(n_size)
        H = np.delete(np.identity(self.N), delate_queue, axis=0)
        Y = np.delete(Y_all, delate_queue, axis=0)
        '''
        logger.debug(n_size)
        logger.debug(Xb.shape)
        logger.debug(Xa.shape)
        logger.debug(H.shape)
        logger.debug(Y.shape)
        logger.debug(H)
        '''
        
        Xa[:, 0] = Y_all[:, 100]

        for t in range(1, step):
            Xb[:, t] = self.RK4(Xa[:, t-1])
            K = (B@H.T)@np.linalg.inv(H@B@H.T + R)
            Xa[:, t] = Xb[:, t] + K@(Y[:,t] - H@Xb[:, t])

        return Xa

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
    
    # RMSEのaveを取得
    def get_RMSE_Ave(self, d, Xas_RMSE):
        rmse_aves = []
        for i in range(len(d)):
            rmse_ave = sum(Xas_RMSE[i]) / len(Xas_RMSE[i])
            rmse_aves.append(rmse_ave)
        return rmse_aves
    
    # RMSEを取得
    def RMSE(self, X1, X2, step):
        rmse = np.zeros((step))
        for i in range(step):
            sub = X1[1:, i] - X2[1:, i]
            rmse[i] = np.sqrt(np.mean(sub**2))
        return rmse
    
    # Spreadを取得
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
            t[j] =self.dt*j*5.0
        return Xn, t
    
    def PartOfdata(self, data_all, delate_queue):
        data = np.delete(data_all, delate_queue, axis=0)
        data = np.delete(data, delate_queue, axis=1)
        return data