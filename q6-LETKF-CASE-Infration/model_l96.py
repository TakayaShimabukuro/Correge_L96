from ctypes.wintypes import HACCEL
import numpy as np
from logging import getLogger, DEBUG, basicConfig
import tqdm
from localization import Localization

logger = getLogger(__name__)

class Model_L96:
    #PARAMETER
    def __init__(self, N, F, dt, inflation, plot):
        self.N = N
        self.F = F
        self.dt = dt
        self.plot = plot
        self.inflation = inflation
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

    def analysis_Xb(self, Xb, Xa, t, ensamble_size):
        for i in range(ensamble_size):
            Xb[:, t, i] = self.RK4(Xa[:, t-1, i])
        return Xb, np.mean(Xb[:, t, :], axis=1)

    def calculate_Zb(self, Xb, Xb_mean, t, ensamble_size):
        dXb = np.zeros((self.N, ensamble_size))
        for i in range(ensamble_size):
            dXb[:, i] = (Xb[:, t, i]-Xb_mean)
        return dXb  * (1+self.inflation) / np.sqrt(ensamble_size-1)

    def calculate_Yb(self, Zb, ensamble_size, delate_queue):
        Yb = np.zeros((self.H.shape[0], ensamble_size))
        for i in range(ensamble_size):
            Yb[:, i] = np.delete(Zb[:, i], delate_queue, axis=0)
        return Yb
        
    def calculate_Pa_tilda(self, Yb, R_loc_inv, I):
        Pa_tilde_inv = I + Yb.T @ R_loc_inv @ Yb
        D, lamda = np.linalg.eigh(Pa_tilde_inv)
        return lamda @ np.diag(1/D) @ lamda.T, lamda @ np.diag(1 / np.sqrt(D)) @ lamda.T
    
    def get_deleate_queue(self, delete_step):
        deleate_queue = []
        
        if delete_step == 1:
            que = np.arange(20, 40, 5)
            for i, data in enumerate(que):
                deleate_queue.append(np.arange(20, data, delete_step))
                logger.debug(deleate_queue[i])
        
        if delete_step >=2:
            que = np.arange(15, 40, 5)
            for i, data in enumerate(que):
                sub_range = 5*i*delete_step
                deleate_queue.append(np.arange(0, sub_range, delete_step))
        
        return deleate_queue

    # LETKF
    def LETKF(self, Y, ensamble_size, step, sigma, delate_queue):

        # PARAMETER
        self.H = np.delete(np.identity(self.N), delate_queue, axis=0)
        h_obs = self.H.shape[0]
        Xb = np.zeros(((self.N , step, ensamble_size)))
        Xa = np.zeros(((self.N , step, ensamble_size)))
        Xb_mean = np.zeros((self.N , step))
        Xa_mean = np.zeros((self.N , step))
        Pb = np.zeros(((self.N, self.N, step)))
        one = np.ones((1, ensamble_size))
        I = np.identity(ensamble_size)
        Yo = np.delete(Y.copy(), delate_queue, axis=0)
        local = Localization(self.N)
        
        # t = 0
        Pb[:, :, 0] = np.diag([25]*self.N)
        for i in range(ensamble_size):
            Xb[:, 0, i] = Y[:, i] + np.random.normal(loc=0.0, scale=1.0, size=self.N) * 2
            Xa[:, 0, i] = Y[:, i]+  np.random.normal(loc=0.0, scale=1.0, size=self.N) * 2

        # t > 0
        for t in tqdm.tqdm(range(1, step), leave=False):
            Xb, Xb_mean = self.analysis_Xb(Xb, Xa, t, ensamble_size) 
            xb_mean_ens = (np.ones((ensamble_size, 1)) @ Xb_mean.reshape(1, -1)).T
            Zb = self.calculate_Zb(Xb, Xb_mean, t, ensamble_size)
            Yb = self.calculate_Yb(Zb, ensamble_size, delate_queue)
            d_ob = Yo[:, t] - self.H @ Xb_mean

            for i in range(self.N):
                L = local.get_L(sigma, self.H, i)
                R_loc_inv = np.diag(np.ones(h_obs) * L)
                Pa_tilde, Pa_tilde_sqrt = self.calculate_Pa_tilda(Yb, R_loc_inv, I)
                
                T = (Pa_tilde @ Yb.T @ R_loc_inv @ d_ob).reshape(-1, 1) @ one + \
                    (np.sqrt(ensamble_size - 1) * Pa_tilde_sqrt)
                Xa[i, t, :] = (xb_mean_ens + (Zb @ T))[i, :]
            Xa_mean[:, t] = np.mean(Xa[:, t, :], axis=1)
            
            Pb[:, :, t] = Zb@I@Zb.T
            '''
            if t %5 == 0:
                logger.debug("Y:\n{}".format(Y[0:3, t]))
                logger.debug("Xb:\n{}".format(Xb[0:3, t, 0]))
                logger.debug("Xa:\n{}".format(Xa[0:3, t, 0]))
                
                logger.debug("Xb_mean:\n{}".format(Xb_mean[0:3, 0:3]))
                logger.debug("Pa_tilde:\n{}".format(Pa_tilde[0:3, 0:3]))
                logger.debug("Pa_tilde_sqrt:\n{}".format(Pa_tilde_sqrt[0:3, 0:3]))
                logger.debug("d_ob:\n{}".format(d_ob[0:3]))
                logger.debug("T:\n{}".format(T[0:3]))
                
                logger.debug("Xa_mean:\n{}".format(Xa_mean[0:3, 0:3]))
                logger.debug("Pb:\n{}".format(Pb[0:3, 0:3, t]))
                '''
            
        
        return Xa, Xa_mean, Pb