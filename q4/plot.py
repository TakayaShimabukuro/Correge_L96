import matplotlib.pyplot as plt
import os
import numpy as np
import scipy as sp
import sympy
from sympy.codegen.cfunctions import log10
from scipy.stats import norm
from scipy.interpolate import interp1d
from logging import getLogger, DEBUG, basicConfig

logger = getLogger(__name__)


class Plot_Methods:
    def __init__(self, path):
        self.path = path
        self.make_file(self.path)
    

    def make_file(self, path):
        try:
            os.mkdir(path)
        except FileExistsError:
            pass        
    

    def VarianceInfration(self, d, t, Xas_RMSE, Pas_Spread):
        x_step=[25]
        x_start=[0]
        x_end=[175]
        x_label = "time(day)"
        y_label = "RMSE"
        line_labels = ["RMSE", "Spread"]
        title = "Lecture4-EKF"
        self.make_file(self.path+"/VarianceInfration")
        for i in range(len(d)):
            for j in range(len(x_step)):
                plt.figure()
                plt.xticks(np.arange(x_start[j], x_end[j], step=x_step[j]))
                plt.plot(t[x_start[j]*4:x_end[j]*4], Xas_RMSE[i][x_start[j]*4:x_end[j]*4], label=line_labels[0])
                plt.plot(t[x_start[j]*4:x_end[j]*4], Pas_Spread[i][x_start[j]*4:x_end[j]*4], label=line_labels[0])
                plt.legend()
                plt.grid(color='k', linestyle='dotted', linewidth=0.5)
                plt.xlim(x_start[j],x_end[j]+1)
                plt.xlabel(x_label)
                plt.ylabel(y_label)
                plt.title(title)
                plt.savefig(self.path + "VarianceInfration/VarianceInfration-delta-" + str(d[i])+ "-day-" + str(x_start[j]) +"-" + str(x_end[j]) + ".png")
                plt.close()
    

    def X1asFuncOfTime(self, d, t, Xt, Y, Xfs, Xas):
        x_step=[1, 10, 10]
        x_start=[0, 0, 250]
        x_end=[5, 50, 300]
        x_label = "time(day)"
        y_label = "X"
        line_labels = ["Truth", "Observe", "Forcast", "Analysis"]
        title = "Lecture4-EKF"
        self.make_file(self.path+"/X1asFuncOfTime")

        for i in range(len(d)):
            for j in range(len(x_step)):
                plt.figure()
                plt.xticks(np.arange(x_start[j], x_end[j], step=x_step[j]))
                plt.plot(t[x_start[j]*4:x_end[j]*4], Xt[1, x_start[j]*4:x_end[j]*4], label=line_labels[0])
                plt.plot(t[x_start[j]*4:x_end[j]*4], Y[1, x_start[j]*4:x_end[j]*4], label=line_labels[1])
                plt.plot(t[x_start[j]*4:x_end[j]*4], Xfs[i][1, x_start[j]*4:x_end[j]*4], label=line_labels[2])
                plt.plot(t[x_start[j]*4:x_end[j]*4], Xas[i][1, x_start[j]*4:x_end[j]*4], label=line_labels[3])
                plt.legend()
                plt.grid(color='k', linestyle='dotted', linewidth=0.5)
                plt.xlim(x_start[j],x_end[j]+1)
                plt.xlabel(x_label)
                plt.ylabel(y_label)
                plt.title(title)
                plt.savefig(self.path + "X1asFuncOfTime/X1asFuncOfTime-delta-" + str(d[i])+ "-day-" + str(x_start[j]) +"-" + str(x_end[j]) + ".png")
                plt.close()
    

    def AnalysisRMSE(self, d, t, Xas_RMSE):
        x_step=[60, 10, 10, 1]
        x_start=[0, 0, 250, 0]
        x_end=[300, 50, 300, 5]
        x_label = "time(day)"
        y_label = "RMSE"
        line_labels = ["delt="+str(d[0]), "delta="+str(d[1]), "delta="+str(d[2]), "delta="+str(d[3])]
        title = "Lecture4-EKF"
        self.make_file(self.path+"/AnalysisRMSE")

        for j in range(len(x_step)):
            plt.figure()
            plt.xticks(np.arange(x_start[j], x_end[j], step=x_step[j]))
            plt.plot(t[x_start[j]*4:x_end[j]*4], Xas_RMSE[0][x_start[j]*4:x_end[j]*4], label=line_labels[0])
            plt.plot(t[x_start[j]*4:x_end[j]*4], Xas_RMSE[1][x_start[j]*4:x_end[j]*4], label=line_labels[1])
            plt.plot(t[x_start[j]*4:x_end[j]*4], Xas_RMSE[2][x_start[j]*4:x_end[j]*4], label=line_labels[2])
            plt.plot(t[x_start[j]*4:x_end[j]*4], Xas_RMSE[3][x_start[j]*4:x_end[j]*4], label=line_labels[3])
            plt.legend()
            plt.grid(color='k', linestyle='dotted', linewidth=0.5)
            plt.xlim(x_start[j],x_end[j]+1)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(title)
            plt.savefig(self.path + "AnalysisRMSE/AnalysisRMSE-day-" + str(x_start[j]) +"-" + str(x_end[j]) + ".png")
            plt.close()


    def VarianceInfrationDelta(self, data, params, names):
        plt.figure()
        self.make_file(names[0])
        plt.xticks(np.arange(params[0], params[1], step=params[2]))
        for i in range(params[3]):
            label = "delt=" + str(names[5][i])
            logger.info(label)
            plt.plot(data[1][params[0]*4+1:params[1]*4+1], data[0][i, params[0]*4+1:params[1]*4+1], label=label)
        plt.grid(color='k', linestyle='dotted', linewidth=0.5)
        plt.xlim(params[0],params[1]-1)
        plt.legend()
        plt.xlabel(names[2])
        plt.ylabel(names[3])
        plt.title(names[4])
        plt.savefig(names[0] + names[1])
        plt.close()

    def VarianceInfrationDeltaPickUp(self, data, params, names):
        plt.figure()
        self.make_file(names[0])
        plt.xticks(np.arange(params[0], params[1], step=params[2]))
        for i in range(params[3]):
            label = "delt=" + str(names[5][i])
            logger.info(label)
            plt.plot(data[1][params[0]*4+1:params[1]*4+1], data[0][i, params[0]*4+1:params[1]*4+1], label=label)
        plt.grid(color='k', linestyle='dotted', linewidth=0.5)
        plt.xlim(params[0],params[1]-1)
        try:
            if params[4]:
                plt.ylim(params[5],params[6])
        except:
            pass

        plt.legend()
        plt.xlabel(names[2])
        plt.ylabel(names[3])
        plt.title(names[4])
        plt.savefig(names[0] + names[1])
        plt.close()
    

    
    
    def aveRMSERatio(self, data, params, names):
        plt.figure()
        self.make_file(names[0])
        plt.plot(data[0], data[1], label="RMSE(analysis-true)", marker="o",fillstyle='none', color="b", linestyle="dashed", lw=0.8)
        plt.legend()
        plt.grid(color='k', linestyle='dotted', linewidth=0.5)
        plt.xlim(params[0],params[1])
        plt.xlabel(names[2])
        plt.ylabel(names[3])
        plt.title(names[4])
        plt.savefig(names[0] + names[1])
        plt.close()