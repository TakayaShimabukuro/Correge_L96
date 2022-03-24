from ctypes.wintypes import HACCEL
import numpy as np
from logging import getLogger, DEBUG, basicConfig
from plot import Plot_Methods
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
        ave_list = []
        for i in range(len(d)):
            ave = sum(data[i]) / len(data[i])
            logger.debug("delta=" + str(d[i]) + "-> RMSE=" + str(ave))
            ave_list.append(ave)
        return ave_list
    
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
    
    def VarianceInflation(self, Xt, Y, d, t_2year, step_t, filePath):
        plot = Plot_Methods()
        step=[1, 10, 10]
        start=[0, 0, 250]
        end=[5, 50, 300]
        for i in range(len(d)):
            for j in range(len(step)):
                # 4. Kalman Filter
                logger.info('Prosess 4')
                Xf, Pf, Xa, Pa = self.KF(Y, d[i])
                fileName = "funcOfTime-" +  str(d[i]) + "-" + str(start[j]) + "-" + str(end[j]) + ".png"
                XLabel = "time(day)"
                YLabel = "X"
                Title = "EKF, funcOfTime"
                data = [Xt, Y, Xf, Xa, t_2year[0:step_t]]
                params = [start[j], end[j]+1, step[j]]
                names = [filePath, fileName, XLabel, YLabel, Title]
                logger.debug('--- params---')
                logger.debug(params)

                plot.funcOfTime(data, params, names)

            # 5. RMSE & Spread
            
            logger.info('Prosess 5')
            Xa_RMSE = self.RMSE(Xa, Xt, step_t)
            Pa_Spread = self.Spread(Pa)

            start = 0
            end = start + 175
            fileName = "varianceInfration-" + str(d[i]) + "-" + str(start) + "-" + str(end) + ".png"
            XLabel = "time(day)"
            YLabel = "X"
            Title = "EKF, VarianceInfration"
            data =[Xa_RMSE, Pa_Spread, t_2year[0:step_t]]
            params = [start, end+1]
            names = [filePath, fileName, XLabel, YLabel, Title]
            plot.VarianceInfration(data, params, names)
    
    def AnalysisRMSE(self, Xt, Y, d, t_2year, step_t, filePath):
        plot = Plot_Methods()
        Xa_deltas = np.zeros((len(d), step_t))
        for i in range(len(d)):
            Xf, Pf, Xa, Pa = self.KF(Y, d[i])
            Xa_RMSE = self.RMSE(Xa, Xt, step_t)
            Xa_deltas[i, :] = Xa_RMSE

        start = 0
        end = 300
        step = 60
        fileName = "AnalysisRMSE-" + str(d[i]) + "-" + str(start) + "-" + str(end) + ".png"
        XLabel = "time(day)"
        YLabel = "RMSE"
        Title = " Lecture4-EKF "
        data = [Xa_deltas, t_2year[0:step_t]]
        params = [start, end+1, step, len(d)]
        names = [filePath, fileName, XLabel, YLabel, Title, d]
        plot.VarianceInfrationDelta(data, params, names)

        start2 = 0
        end2 = 20
        step2 = 2.5
        fileName2 = "AnalysisRMSE-" + \
        str(d[i]) + "-" + str(start2) + "-" + str(end2) + ".png"
        params = [start2, end2+1, step2, len(d), True, 0, 1.4]
        names = [filePath, fileName2, XLabel, YLabel, Title, d]
        plot.VarianceInfrationDeltaPickUp(data, params, names)

        start3 = 220
        end3 = 300
        step3 = 10
        fileName3 = "AnalysisRMSE-" + \
        str(d[i]) + "-" + str(start3) + "-" + str(end3) + ".png"
        params = [start3, end3+1, step3, len(d), True, 0, 1.4]
        names = [filePath, fileName3, XLabel, YLabel, Title, d]
        plot.VarianceInfrationDeltaPickUp(data, params, names)

        return Xa_deltas
    
    def RatioRMSE(self, Xa_deltas, Xt, Y, d, t_2year, step_t, filePath):
        plot = Plot_Methods()
        start = 0
        end = 0.225
        step = 0.025
        fileName3 = "RatioRMSE.png"
        XLabel = "Infration Ratio"
        YLabel = "ave RMSE"
        Title = " Lecture4-EKF "
        ave_list = self.showAveRMSE(Xa_deltas, d)
        data = [d, ave_list, t_2year[0:step_t]]
        params = [start, end, step, len(d), True, 0, 1.4]
        names = [filePath, fileName3, XLabel, YLabel, Title, d]
        plot.VarianceInfrationDeltaPickUp(data, params, names)


