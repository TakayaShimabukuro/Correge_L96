from ctypes.wintypes import HACCEL
import numpy as np
from logging import getLogger, DEBUG, basicConfig
from plot import Plot_Methods
logger = getLogger(__name__)


class Model_L96:
    #PARAMETER
    def __init__(self, N, F, dt, delta, plot):
        self.N = N
        self.F = F
        self.dt = dt
        self.plot = plot
        self.delta = delta
        self.H = np.identity(self.N)
        self.R = np.identity(self.N)


    # METHOD
    def analyze_model(self, Xn, X1_tmp, step):    # X1[40, 1]を初期値とするstep分のシミュレーションを行う。
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

    def l96(self, x):
        f = np.zeros_like(x)
        f[0] = (x[1]-x[self.N-2])*x[self.N-1]-x[0]+self.F
        f[self.N-1] = (x[0]-x[self.N-3])*x[self.N-2]-x[self.N-1]+self.F
        for i in range(1, self.N-1):
            f[i] = (x[i+1]-x[i-2])*x[i-1]-x[i]+self.F
        return f

    def RK4(self, X1):
        k1 = self.l96(X1) * self.dt
        k2 = self.l96(X1 + k1*0.5) * self.dt
        k3 = self.l96(X1 + k2*0.5) * self.dt
        k4 = self.l96(X1 + k3) * self.dt
        return X1 + (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0

    def RMSE(self, X1, X2, step):
        rmse = np.zeros((step))
        for t in range(step):
            sub = X1[:, t] - X2[:, t]
            rmse[t] = np.sqrt(np.mean(sub**2))
        return rmse

    def Spread(self, P, step):
        tmp = []
        for t in range(step):
            tmp.append(np.sqrt(np.trace(P[:, :, t])/len(P[:, :, t])))
        return tmp

    def generete_noises_avarage_0(self, ensamble_size):
        noise_list = []
        for i in range(ensamble_size):
            noise_list.append(np.random.normal(loc=0.0, scale=1.0, size=self.N))
        return noise_list, np.mean(noise_list)

    def analysis_Xb(self, Xb, Xa, t, ensamble_size):
        for i in range(ensamble_size):
            Xb[:, t, i] = self.RK4(Xa[:, t-1, i])
        return Xb, np.mean(Xb[:, t, :], axis=1)

    def analysis_Xa(self, Xb, Xa, K, Y, t, ensamble_size):
        noise_list, noise_ave = self.generete_noises_avarage_0(ensamble_size)
        for i in range(ensamble_size):
            Xa[:, t, i] = Xb[:, t, i] + K@(Y[:, t] + (noise_list[i]-noise_ave) - self.H@Xb[:, t, i])
        return Xa, np.mean(Xa[:, t, :], axis=1)

    def calculate_Zb(self, Xb, Xb_mean, t, ensamble_size):
        dXb = np.zeros((self.N, ensamble_size))
        for i in range(ensamble_size):
            dXb[:, i] = (Xb[:, t, i]-Xb_mean)
        return (dXb / np.sqrt(ensamble_size-1)) * (1 + self.delta)

    def calculate_Yb(self, Zb, ensamble_size):
        Yb = np.zeros((self.N, ensamble_size))
        for i in range(ensamble_size):
            Yb[:, i] = self.H @ Zb[:, i]
        return Yb

    # PO法によるEnKF
    def EnKF_PO(self, Y, ensamble_size, step, L):

        # PARAMETER
        Xb = np.zeros(((self.N, step, ensamble_size)))
        Xa = np.zeros(((self.N, step, ensamble_size)))
        Xb_mean = np.zeros((self.N, step))
        Xa_mean = np.zeros((self.N, step))
        Pb = np.zeros(((self.N, self.N, step)))

        # t = 0
        Pb[:, :, 0] = np.diag([25]*self.N)
        for i in range(ensamble_size):
            Xb[:, 0, i] = Y[:, i]+np.random.normal(loc=0.0, scale=1.0, size=self.N)
            Xa[:, 0, i] = Y[:, i]+np.random.normal(loc=0.0, scale=1.0, size=self.N)

        # t > 0
        for t in range(1, step):
            Xb, Xb_mean[:, t] = self.analysis_Xb(Xb, Xa, t, ensamble_size)
            Zb = self.calculate_Zb(Xb, Xb_mean[:, t], t, ensamble_size)
            Yb = self.calculate_Yb(Zb, ensamble_size)
            K = L * (Zb@Yb.T@np.linalg.inv((self.H@L)*(Yb@Yb.T) + self.R))
            Xa, Xa_mean[:, t] = self.analysis_Xa(Xb, Xa, K, Y, t, ensamble_size)
            Pb[:, :, t] = Zb@Zb.T
        
        return Xa, Xa_mean, Pb