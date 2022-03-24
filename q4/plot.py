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
    

    def X1asFuncOfTime(self, d, t, Xt, Y, Xfs, Xas):
        x_step=[1, 10, 10]
        x_start=[0, 0, 250]
        x_end=[6, 51, 301]
        x_label = "time(day)"
        y_label = "X"
        line_labels = ["Truth", "Observe", "Forcast", "Analysis"]
        title = "Lecture4-EKF"
        self.make_file(self.path+"/X1asFuncOfTime")

        for i in range(len(d)):
            for j in range(len(x_step)):
                plt.figure()
                plt.xticks(np.arange(x_start[j], x_end[j], step=x_step[j]))
                plt.plot(t[x_start[j]*4:x_end[j]*4+x_step[j]], Xt[1, x_start[j]*4:x_end[j]*4+x_step[j]], label=line_labels[0])
                plt.plot(t[x_start[j]*4:x_end[j]*4+x_step[j]], Y[1, x_start[j]*4:x_end[j]*4+x_step[j]], label=line_labels[1])
                plt.plot(t[x_start[j]*4:x_end[j]*4+x_step[j]], Xfs[i][1, x_start[j]*4:x_end[j]*4+x_step[j]], label=line_labels[2])
                plt.plot(t[x_start[j]*4:x_end[j]*4+x_step[j]], Xas[i][1, x_start[j]*4:x_end[j]*4+x_step[j]], label=line_labels[3])
                plt.legend()
                plt.grid(color='k', linestyle='dotted', linewidth=0.5)
                plt.xlim(x_start[j],x_end[j]-1)
                plt.xlabel(x_label)
                plt.ylabel(y_label)
                plt.title(title)
                plt.savefig(self.path + "X1asFuncOfTime/X1asFuncOfTime-delta-" + str(d[i])+ "-day-" + str(x_start[j]) +"-" + str(x_end[j]-1) + ".png")
                plt.close()
    

    def AnalysisRMSE(self, d, t, Xas_RMSE):
        x_step=[60, 10, 10, 1]
        x_start=[0, 0, 250, 0]
        x_end=[301, 51, 301, 6]
        x_label = "time(day)"
        y_label = "RMSE"
        line_labels = ["delt="+str(d[0]), "delta="+str(d[1]), "delta="+str(d[2]), "delta="+str(d[3]), "delta="+str(d[4])]
        title = "Lecture4-EKF"
        self.make_file(self.path+"/AnalysisRMSE")

        for j in range(len(x_step)):
            plt.figure()
            plt.xticks(np.arange(x_start[j], x_end[j], step=x_step[j]))
            plt.plot(t[x_start[j]*4:x_end[j]*4+x_step[j]], Xas_RMSE[0][x_start[j]*4:x_end[j]*4+x_step[j]], label=line_labels[0])
            plt.plot(t[x_start[j]*4:x_end[j]*4+x_step[j]], Xas_RMSE[1][x_start[j]*4:x_end[j]*4+x_step[j]], label=line_labels[1])
            plt.plot(t[x_start[j]*4:x_end[j]*4+x_step[j]], Xas_RMSE[2][x_start[j]*4:x_end[j]*4+x_step[j]], label=line_labels[2])
            plt.plot(t[x_start[j]*4:x_end[j]*4+x_step[j]], Xas_RMSE[3][x_start[j]*4:x_end[j]*4+x_step[j]], label=line_labels[3])
            plt.plot(t[x_start[j]*4:x_end[j]*4+x_step[j]], Xas_RMSE[3][x_start[j]*4:x_end[j]*4+x_step[j]], label=line_labels[4])
            plt.legend()
            plt.grid(color='k', linestyle='dotted', linewidth=0.5)
            plt.xlim(x_start[j],x_end[j]-1)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(title)
            plt.savefig(self.path + "AnalysisRMSE/AnalysisRMSE-day-" + str(x_start[j]) +"-" + str(x_end[j]-1) + ".png")
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