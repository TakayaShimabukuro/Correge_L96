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
            sub = X1[:, 0, t] - X2[:, 0, t]
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
    def EnKF_PO(self, Y, m, noise, step):
        m_len = len(m)
        Xa = np.zeros(((self.N, self.N, step)))
        Pa = np.zeros(((self.N, self.N, step)))
        Xbs = []
        Xas = []

        for i in range(m_len):
            Xbs.append(np.zeros(((self.N, self.N, step))))
            Xas.append(np.zeros(((self.N, self.N, step))))
            Xbs[i][:, 0, 0] = Y[:, 0, m[i]]+noise
            Xas[i][:, 0, 0] = Y[:, 0, m[i]]+noise

        for t in range(1, step):
            # init setting
            H = np.identity(self.N)
            R = np.identity(self.N)
            I = np.identity(self.N)
            K = np.zeros((self.N, self.N))
            Xb_sum = np.zeros((self.N, self.N))
            Xa_sum = np.zeros((self.N, self.N))
            dXb = np.zeros((self.N, self.N))
            Zb = np.zeros((self.N, self.N))
            Yb = np.zeros((self.N, self.N))
            

            for i in range(m_len):
                # 1 Ensemble Prediction (state)
                Xbs[i][:, 0, t] = self.RK4(Xas[i][:, 0, t-1])
                Xb_sum[:, 0] = np.add(Xb_sum[:, 0], Xbs[i][:, 0, t])
            
            for i in range(m_len):
                dXb[:, 0] = Xbs[i][:, 0, t]-(Xb_sum[:, 0] / m_len)
                Zb[:, 0]  = (dXb[:, 0] / np.sqrt(m_len-1))
                Yb = H@Zb

                # 3 Kalman Gain
                K[:, :] = Zb@Yb.T@np.linalg.inv((Yb@Yb.T)+R)

                # 4 Analysis (state)
                Xas[i][:, 0, t] = Xbs[i][:, 0, t] + K@(Y[:, 0, t]+noise-H@Xbs[i][:, 0, t])
                Xa_sum[:, 0] = Xa_sum[:, 0] + Xas[i][:, 0, t]

                if(i < 1):
                    logger.info("------------------------------------------------")
                    logger.debug('dXb[0:5, 0:5]\n{}'.format(dXb[0:5, 0:5]))
                    logger.debug('Zb[0:5, 0:5]\n{}'.format(Zb[0:5, 0:5]))
                    logger.debug('Yb[0:5, 0:5]\n{}'.format(Yb[0:5, 0:5]))
                    logger.debug('K[0:3, 0:3]\n{}'.format(K[0:3, 0:3]))
                    logger.debug('Xbs[0][0:5, 0:5, t]\n{}'.format(Xbs[i][0:5, 0:5, t]))
                    logger.debug('Xas[0][0:5, 0:5, t]\n{}'.format(Xas[i][0:5, 0:5, t]))
            
            Xa[:, 0, t] = Xa_sum[:, 0] / m_len
            
            
            # Pa
            Pa_sum = np.zeros((self.N, self.N))
            dXa = np.zeros((self.N, self.N))
            Za = np.zeros((self.N, self.N))
            
            for i in range(m_len):
                dXa[:, 0] = Xas[i][:, 0, t] - (Xa_sum[:, 0] / m_len)
                Za[:, 0] = dXa[:, 0]/np.sqrt(m_len-1)
                Pa_sum = np.add(Za@Za.T, Pa_sum)

            Pa[:, :, t] = Pa_sum / m_len

            if(t < 2):
                logger.info("------------------------------------------------")
                logger.debug('Pa_sum[:, :]\n{}'.format(Pa_sum[0:5, 0:5]))
                logger.debug('Pa[0:5, 0:5, t]\n{}'.format(Pa[0:5, 0:5, t]))
                logger.debug('dXa[:, 0]\n{}'.format(dXa[0:5, 0:5]))
                logger.debug('Za[:, 0] \n{}'.format(Za[0:5, 0:5] ))
                #self.plot.Debug(Ybs[0], "Ybs[0]")
                #self.plot.Debug(Ybs[1], "Ybs[1]")
            
            
        return Xa, Pa
