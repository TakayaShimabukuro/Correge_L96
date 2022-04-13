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
        for i in range(step):
            sub = X1[1:, i] - X2[1:, i]
            rmse[i] = np.sqrt(np.mean(sub**2))
        return rmse

    # Spreadを取得
    def Spread(self, P, step):
        tmp = []
        for t in range(step):
            tmp.append(np.sqrt(np.trace(P[:, :, t])/len(P[:, :, t])))
        return tmp

    # PO法によるEnKF
    def EnKF_PO(self, Y, m, noise):
        # init setting
        step = len(Y[0])
        m_len = len(m)
        Xa = np.zeros((self.N, step))
        Pa = np.zeros((self.N, self.N, step))
        H = np.identity(self.N)
        R = np.identity(self.N)
        I = np.identity(self.N)

        # m個格納用
        Xbs = []
        Xas = []
        Pbs = []
        Pas = []

        # 1-0 準備 OK
        for i in range(m_len):
            Xbs.append(np.zeros((self.N, step)))
            Pbs.append(np.zeros(((self.N, self.N, step))))
            Xas.append(np.zeros((self.N, step)))
            Pas.append(np.zeros(((self.N, self.N, step))))

        for i in range(m_len):
            Xbs[i][:, 0] = Y[:, m[i]]+noise+10
            Pbs[i][:, :, 0] = np.diag([m[i]]*self.N)
            Xas[i][:, 0] = Y[:, m[i]]+noise+10
            Pas[i][:, :, 0] = np.diag([m[i]]*self.N)
            logger.debug('Y[:, m[i]]\n{}'.format(Y[:, m[i]]))

        for t in range(1, step):
            Xb_sum = np.zeros((self.N, self.N))
            Xa_sum = np.zeros((self.N, self.N))

            for i in range(m_len):
                # 1 Ensemble Prediction (state)
                Xbs[i][:, t] = self.RK4(Xas[i][:, t-1])
                Xb_sum[:, 0] = Xb_sum[:, 0] + Xbs[i][:, t]

            for i in range(m_len):
                dXb = np.zeros((self.N, self.N))
                Zb = np.zeros((self.N, self.N))
                Yb = np.zeros((self.N, self.N))

                dXb[:, 0] = Xbs[i][:, t] - (Xb_sum[:, 0] / m_len)

                Zb[:, 0]  = dXb[:, 0] /np.sqrt(m_len-1)
                Yb = H@Zb

                # 2 Prediction of Error Covariance (implicitly)
                Pbs[i][:, :, t] = Zb@Zb.T

                # 3 Kalman Gain
                K = Zb@np.linalg.inv(I+Yb.T@np.linalg.inv(R)@ Yb)@Yb.T@np.linalg.inv(R)

                # 4 Analysis (state)
                Xas[i][:, t] = Xbs[i][:, t] + K@(Y[:, t] + noise - H@Xbs[i][:, t])
                Xa_sum[:, 0] = Xa_sum[:, 0] + Xas[i][:, t]

                logger.debug('Xbs[i][:, t]\n{}'.format(Xbs[i][:, t] ))
                logger.debug('dXb\n{}'.format(dXb[0:3, 0:3]))
                logger.debug('Pbs[i][:, :, t]\n{}'.format(Pbs[i][0:3, 0:3, t]))
                logger.debug('Zb\n{}'.format(Zb[0:3, 0:3]))
                logger.debug('Yb\n{}'.format(Yb[0:3, 0:3]))
                logger.debug('K\n{}'.format(K[0:3, 0:3]))
                logger.debug('Xa_sum[:, 0]\n{}'.format(Xa_sum[0:3, 0:3]))

            Xa[:, t] = Xa_sum[:, 0] / m_len

            # Pa
            Pa_sum = np.zeros((self.N, self.N))
            for i in range(m_len):
                dXa = np.zeros((self.N, self.N))
                Za = np.zeros((self.N, self.N))

                dXa[:, 0] = Xas[i][:, t] - (Xa_sum[:, 0] / m_len)
                Za[:, 0] = dXa[:, 0]/np.sqrt(m_len-1)

                Pas[i][:, :, t] = Za@Za.T
                Pa_sum = Pa_sum + Pas[i][:, :, t]

            Pa[:, :, t] = Pa_sum[:, :] / m_len

            logger.debug('Xa[:, %d]\n{}'.format(Xa[:, t], t))
            logger.debug('Pa[:, :, %d]\n{}'.format( Pa[:, :, t], t))

            
        return Xa, Pa

        #logger.debug('Ybs[0][0, 0]={}'.format(Ybs[0][0:3, 0:3]))
        #logger.debug('Ybs[1][0, 0]={}'.format(Ybs[1][0:3, 0:3]))
        #self.plot.Debug(Ybs[0], "Ybs[0]")
        #self.plot.Debug(Ybs[1], "Ybs[1]")
