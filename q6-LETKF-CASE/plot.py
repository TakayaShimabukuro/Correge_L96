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
    def __init__(self, path, title):
        self.title = title
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
    def TimeMeanRMSEEnsembles(self, L, ensamble_size, Xas_RMSE):
        self.make_file(self.path+"/TimeMeanRMSEEnsembles")
        plt.figure()
        for k in range(len(ensamble_size)):
            plt.plot(L, Xas_RMSE[k], marker="o",fillstyle='none', linestyle="dashed", lw=0.8, label="ensamble_size:" + str(k))
        plt.grid(color='k', linestyle='dotted', linewidth=0.5)
        plt.xlabel("sigma")
        plt.ylabel("RMSE")
        plt.legend()
        plt.title(self.title)
        plt.savefig(self.path + "TimeMeanRMSEEnsembles/TimeMeanRMSEEnsembles.png")
        plt.close()

    def TimeMeanRMSE(self, B_step, Xas_RMSE):
        self.make_file(self.path+"/TimeMeanRMSE")
        plt.figure()
        plt.plot(B_step, Xas_RMSE, marker="o",fillstyle='none', color="b", linestyle="dashed", lw=0.8)
        plt.grid(color='k', linestyle='dotted', linewidth=0.5)
        plt.xlabel("sigma")
        plt.ylabel("RMSE")
        plt.title(self.title)
        plt.savefig(self.path + "TimeMeanRMSE/TimeMeanRMSE.png")
        plt.close()

    def AnalysisRMSEandTrace(self, t, RMSE, Trace, name):
        self.make_file(self.path+"/AnalysisRMSEandTrace")
        plt.figure()
        plt.xticks(np.arange(0, 364, step=50))
        plt.plot(t[0*4:364*4], RMSE[0*4:364*4], label="Xa_RMSE")
        plt.plot(t[0*4:364*4], Trace[0*4:364*4], label="Pb_Trace")
        plt.legend()
        plt.grid(color='k', linestyle='dotted', linewidth=0.5)
        plt.xlim(0, 364-1)
        plt.xlabel("time(day)")
        plt.ylabel("RMSE")
        plt.title(self.title)
        plt.savefig(self.path + "AnalysisRMSEandTrace/AnalysisRMSEandTrace-" + name + ".png")
        plt.close()


    def SensitivityTOObs(self, LETKF):
        self.make_file(self.path+"/SensitivityTOObs")
        data_range = np.arange(15, 40, 5)
        x_label = "Observations"
        y_label = "best RMSE"
        title = "Lecture5-3DVAR"
        j = 0
        plt.figure()
        plt.plot(data_range, LETKF, marker="o",fillstyle='none', color="b", linestyle="dashed", lw=0.8, label="KF")
        plt.grid(color='k', linestyle='dotted', linewidth=0.5)
        plt.legend()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.savefig(self.path + "SensitivityTOObs/SensitivityTOObs.png")
        plt.close()