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

    def TimeMeanRMSE(self, B_step, Xas_RMSE):
        self.make_file(self.path+"/TimeMeanRMSE")
        x_label = "sigma"
        y_label = "ave RMSE(Time mean)"
        title = "Lecture6-EnKF"
        plt.figure()
        plt.plot(B_step, Xas_RMSE, marker="o",fillstyle='none', color="b", linestyle="dashed", lw=0.8)
        plt.grid(color='k', linestyle='dotted', linewidth=0.5)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.savefig(self.path + "TimeMeanRMSE/TimeMeanRMSE.png")
        plt.close()
    
