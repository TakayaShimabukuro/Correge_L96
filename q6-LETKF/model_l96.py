from ctypes.wintypes import HACCEL
import numpy as np
from logging import getLogger, DEBUG, basicConfig
from plot import Plot_Methods
logger = getLogger(__name__)


class Model_L96:
    # 初期値
    def __init__(self, N, F, dt, delta, plot):
        self.N = N
        self.F = F
        self.dt = dt
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

    # ETKF
    def ETKF(self, Y, m_temp, step):

        # PARAMETER
        m_len = len(m_temp)
        Xb = np.zeros(((self.N, step, m_len)))
        Xa = np.zeros(((self.N, step, m_len)))
        Xa_mean = np.zeros((self.N, step))
        Pa = np.zeros(((self.N, self.N, step)))
        H = np.identity(self.N)
        R = np.identity(self.N)
        I = np.identity(m_len)
        for m in range(m_len):
            Xb[:, 0, m] = Y[:, m_temp[m]]+(np.random.normal(loc=0.0, scale=1.0, size=self.N))
            Xa[:, 0, m] = Y[:, m_temp[m]]+(np.random.normal(loc=0.0, scale=1.0, size=self.N))
        Pa[:, :, 0] = np.diag([25]*self.N)
        
        # PROCESS
        for t in range(1, step):

            # PARAMETER that is necessary to update.
            Xb_sum = np.zeros(self.N)
            Xa_sum = np.zeros(self.N)
            dXb = np.zeros((self.N, m_len))
            dXa = np.zeros((self.N, m_len))

            # Background Step
            for m in range(m_len):
                Xb[:, t, m] = self.RK4(Xa[:, t-1, m])
                Xb_sum += Xb[:, t, m]
            
            Xb_mean = Xb_sum / m_len
            
            for m in range(m_len):
                dXb[:, m] = Xb[:, t, m]-Xb_mean
            Zb = dXb / np.sqrt(m_len-1)
            Yb = H@Zb
            Pb_tilde = I

            # Analysis Step
            Pa_tilde = np.linalg.inv(np.linalg.inv(Pb_tilde) + Yb.T@np.linalg.inv(R)@Yb)
            d_ob = Y[:, t] - H@Xb_mean
            #Pa_tilde = np.linalg.inv(I + Yb.T@R@Yb)
            T = (Pa_tilde@Yb.T@R@d_ob*1)+(np.sqrt(m-1)*np.sqrt(Pa_tilde))
            ZbT = Zb@T

            '''
            logger.info("--- Debug1 ---")
            logger.debug("d_ob.shape:\n{}".format(d_ob.shape))
            logger.debug("d_ob:\n{}".format(d_ob[0:3]))
            logger.debug("Pa_tilde.shape:\n{}".format(Pa_tilde.shape))
            logger.debug("Pa_tilde:\n{}".format(Pa_tilde[0:3, 0:3]))
            logger.debug("T.shape:\n{}".format(T.shape))
            logger.debug("T:\n{}".format(T[0:3, 0:3]))
            logger.debug("ZbT.shape:\n{}".format(ZbT.shape))
            logger.debug("ZbT:\n{}".format(ZbT[0:3, 0:3]))
            '''

            for m in range(m_len):
                Xa[:, t, m] = Xb[:, t, m] + ZbT[:, m]
                Xa_sum += Xa[:, t, m]
            
            Xa_mean[:, t] = Xa_sum/ m_len

            for m in range(m_len):
                dXa[:, m] = Xa[:, t, m] - Xa_mean[:, t]
            Za = dXa/np.sqrt(m_len-1)
            Pa[:, :, t] = Za@Pa_tilde@Za.T

        return Xa, Xa_mean, Pa