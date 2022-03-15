import matplotlib.pyplot as plt
import os
import numpy as np
import scipy as sp
import sympy
from sympy.codegen.cfunctions import log10

class Plot_Methods:
    def make_file(self, file_path):
        try:
            os.mkdir(file_path)
        except FileExistsError:
            pass

    def xy_graph_rmse(self, rmse, file_path, tn_forcast, std):
        fig = plt.figure()
        self.make_file(file_path)

        plt.xticks(np.arange(0, 101, step=10))
        plt.plot(tn_forcast, rmse)
        plt.grid(color='k', linestyle='dotted', linewidth=0.5)
        file_name = file_path + str(std) + "-100days" + ".png"
        plt.savefig(file_name)
        plt.close()
    
    
    def xy_graph_rmse_log(self, rmse, file_path, tn_forcast, std):
        fig = plt.figure()
        self.make_file(file_path)
        ax1 = plt.gca() 
        ax2 = ax1.twinx()
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        rmse_log = sp.log10(rmse)
        ax1.plot(tn_forcast, rmse_log, color="orangered")
        ax1.set_ylim([-5, 1])
        ax1.set_ylabel(r"$\log_{10}(RMSE)$")
        ax1.ticklabel_format(style='sci',axis='y',scilimits=(0,0)) 

        ax2.plot(tn_forcast, rmse)
        ax2.set_ylim(-2, 10)
        plt.title("Average Prediction error")
        plt.legend(loc='lower right',fontsize=10)
        plt.xticks(np.arange(0., 50.1, step=10))
        plt.grid(color='k', linestyle='dotted', linewidth=0.5)
        ax1.grid()
        plt.minorticks_on()
        ax1.minorticks_on()
        plt.savefig(file_path + str(std) + "-50days" + ".png")
        plt.close()