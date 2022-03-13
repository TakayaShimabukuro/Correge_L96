import matplotlib.pyplot as plt
import os
import numpy as np

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
        ax1 = fig.subplots()
        ax2 = ax1.twinx()

        ax1.plot(tn_forcast, rmse)
        plt.xticks(np.arange(0, 101, step=10))
        
        ax2.plot(tn_forcast, rmse)
        ax2.set_yscale('log')
        
        plt.grid(color='k', linestyle='dotted', linewidth=0.5)

        file_name = file_path + str(std) + "-50days" + ".png"
        plt.savefig(file_name)
        plt.close()