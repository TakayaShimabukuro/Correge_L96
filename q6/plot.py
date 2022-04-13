import matplotlib.pyplot as plt
import os
import numpy as np
import scipy as sp
import sympy
from sympy.codegen.cfunctions import log10
from scipy.stats import norm
from scipy.interpolate import interp1d
from logging import getLogger, DEBUG, basicConfig
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, BoundaryNorm, rgb2hex
import re

logger = getLogger(__name__)


class Plot_Methods:
    def __init__(self, path):
        self.path = path
        self.make_file(self.path)
    
    def get_color_code(self, cname,num):

        cmap = cm.get_cmap(cname,num)

        code_list =[]
        for i in range(cmap.N):
            rgb = cmap(i)[:3]
            print(rgb2hex(rgb))
            code_list.append(rgb2hex(rgb))

        return code_list
    
    def createCorlorMap(self, color, bounds):
        color_code = self.get_color_code(color,len(bounds)+1)
        return ListedColormap(color_code)

    def make_file(self, path):
        try:
            os.mkdir(path)
        except FileExistsError:
            pass        
    

    def VarianceInfration(self, d, t, Xas_RMSE, Pas_Spread):
        x_step=[25]
        x_start=[0]
        x_end=[176]
        x_label = "time(day)"
        y_label = "RMSE"
        line_labels = ["RMSE", "Spread"]
        title = "Lecture4-EKF"
        self.make_file(self.path+"/VarianceInfration")
        for i in range(len(d)):
            for j in range(len(x_step)):
                plt.figure()
                plt.xticks(np.arange(x_start[j], x_end[j], step=x_step[j]))
                plt.plot(t[x_start[j]*4:x_end[j]*4+x_step[j]], Xas_RMSE[i][x_start[j]*4:x_end[j]*4+x_step[j]], label=line_labels[0])
                plt.plot(t[x_start[j]*4:x_end[j]*4+x_step[j]], Pas_Spread[i][x_start[j]*4:x_end[j]*4+x_step[j]], label=line_labels[1])
                plt.legend()
                plt.grid(color='k', linestyle='dotted', linewidth=0.5)
                plt.xlim(x_start[j],x_end[j]-1)
                plt.xlabel(x_label)
                plt.ylabel(y_label)
                plt.title(title)
                plt.savefig(self.path + "VarianceInfration/VarianceInfration-delta-" + str(d[i])+ "-day-" + str(x_start[j]) +"-" + str(x_end[j]-1) + ".png")
                plt.close()
    

    def X1asFuncOfTime(self, t, Xt, Y, Xa):
        x_step=[10]
        x_start=[0]
        x_end=[100]
        x_label = "time(day)"
        y_label = "X"
        line_labels = ["Truth", "Observe", "Analysis"]
        title = "Lecture6-EnKF-PO"
        self.make_file(self.path+"/EnKF-PO")


        for j in range(len(x_step)):
            plt.figure()
            plt.xticks(np.arange(x_start[j], x_end[j], step=x_step[j]))
            plt.plot(t[x_start[j]*4:x_end[j]*4+x_step[j]], Xt[1, x_start[j]*4:x_end[j]*4+x_step[j]], label=line_labels[0])
            plt.plot(t[x_start[j]*4:x_end[j]*4+x_step[j]], Y[1, x_start[j]*4:x_end[j]*4+x_step[j]], label=line_labels[1])
            plt.plot(t[x_start[j]*4:x_end[j]*4+x_step[j]], Xa[1, x_start[j]*4:x_end[j]*4+x_step[j]], label=line_labels[2])
            plt.legend()
            plt.grid(color='k', linestyle='dotted', linewidth=0.5)
            plt.xlim(x_start[j],x_end[j]-1)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(title)
            plt.savefig(self.path + "EnKF-PO/EnKF-PO-" + str(x_start[j]) +"-" + str(x_end[j]-1) + ".png")
            plt.close()
    
    

    def AnalysisRMSEandTrace(self, t, RMSE, Trace):
        x_step=[50]
        x_start=[0] # day
        x_end=[364] # day
        x_label = "time(day)"
        y_label = "RMSE"
        title = "Lecture6-EKF"
        self.make_file(self.path+"/AnalysisRMSEandTrace")

        for j in range(len(x_step)):
            plt.figure()
            plt.xticks(np.arange(x_start[j], x_end[j], step=x_step[j]))
            plt.plot(t[x_start[j]*4:x_end[j]*4], RMSE[x_start[j]*4:x_end[j]*4], label="Xa_RMSE")
            plt.plot(t[x_start[j]*4:x_end[j]*4], Trace[x_start[j]*4:x_end[j]*4], label="Xa_Trace")
            plt.legend()
            plt.grid(color='k', linestyle='dotted', linewidth=0.5)
            plt.xlim(x_start[j],x_end[j]-1)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(title)
            plt.savefig(self.path + "AnalysisRMSEandTrace/AnalysisRMSEandTrace-" + str(x_start[j]) +"-" + str(x_end[j]-1) + ".png")
            plt.close()

    
    def InflationRatio(self, d, rmse_aves):
        self.make_file(self.path+"/InflationRatio")
        x_step=[0.025]
        x_start=[0.00]
        x_end=[0.225]
        x_label = "Inflation Ratio"
        y_label = "ave RMSE"
        line_labels = ["RMSE(analysis-true)"]
        title = "Lecture4-EKF"
        j = 0
        plt.figure()
        plt.xticks(np.arange(x_start[j], x_end[j], step=x_step[j]))
        plt.plot(d, rmse_aves, label=line_labels[j], \
            marker="o",fillstyle='none', color="b", linestyle="dashed", lw=0.8)
        plt.legend()
        plt.grid(color='k', linestyle='dotted', linewidth=0.5)
        plt.xlim(x_start[j],x_end[j]-x_step[j])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.savefig(self.path + "InflationRatio/InflationRatio.png")
        plt.close()
    

    def AnalysisErrCovariance(self, Pa):
        x_step=[5]
        x_start=[0]
        x_end=[41]
        day = [1, 10, 300]
        x_label = "time(day)"
        y_label = "RMSE"
        title = "Lecture4-EKF"
        self.make_file(self.path+"/AnalysisErrCovariance")

        j = 0
        bounds = [-0.5, -0.3, -0.1, -0.05, -0.01, 0, 0.01, 0.05, 0.1, 0.3, 0.5]
        cmap = self.createCorlorMap("coolwarm", bounds)
        norm = BoundaryNorm(bounds,cmap.N)

        for k in range(len(day)):
            fig = plt.figure()
            plt.xticks(np.arange(x_start[j], x_end[j], step=x_step[j]))
            sc = plt.imshow(Pa[:, :, day[k]*4], cmap=cmap,norm=norm, vmin=min(bounds),vmax=max(bounds))
            plt.legend()
            plt.colorbar(sc,aspect=30,shrink=0.7)
            plt.grid(color='k', linestyle='dotted', linewidth=0.5)
            plt.xlim(x_start[j],x_end[j]-1)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(title)
            plt.savefig(self.path + "AnalysisErrCovariance/AnalysisErrCovariance-delta-day-"+str(day[k])+ ".png")
            plt.close()
    
    def TimeMeanRMSE(self, B_step, Xas_RMSE):
        self.make_file(self.path+"/TimeMeanRMSE")
        x_step=[0.1]
        x_start=[0.00]
        x_end=[0.7]
        y_step=[0.5]
        y_start=[0.00]
        y_end=[3.2]
        x_label = "B"
        y_label = "ave RMSE(Time mean)"
        title = "Lecture5-3DVAR"
        j = 0
        plt.figure()
        plt.plot(B_step, Xas_RMSE, marker="o",fillstyle='none', color="b", linestyle="dashed", lw=0.8)
        plt.grid(color='k', linestyle='dotted', linewidth=0.5)
        plt.xlim(x_start[j],x_end[j]-x_step[j])
        plt.ylim(y_start[j],y_end[j]-y_step[j])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.savefig(self.path + "TimeMeanRMSE/TimeMeanRMSE.png")
        plt.close()
    
    def Debug(self, data, name):
        self.make_file(self.path+"/Debug")
        MAX = np.amax(data)
        MIN = abs(np.amin(data))

        RANGE = 0
        if(MAX>MIN):
            RANGE = MAX
        else:
            RANGE = MIN
        RANGE = round(RANGE, 1-len(str(RANGE).split('.')[0]))
        n = RANGE/5
        bounds =  [-RANGE, -n*4, -n*3, -n*2, -n, -0.1, 0.1, n, n*2, n*3, n*4, RANGE]
        cmap = self.createCorlorMap("bwr", bounds)
        norm = BoundaryNorm(bounds,cmap.N)

        plt.figure()
        plt.imshow(data,cmap=cmap,norm=norm, vmin=min(bounds),vmax=max(bounds))
        plt.colorbar()
        plt.savefig(self.path + "Debug/" + name + ".png")
        plt.close()

