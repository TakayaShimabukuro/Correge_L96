import matplotlib.pyplot as plt
from logging import getLogger
import os
import numpy as np
logger = getLogger(__name__)



class Plot_Methods:
    def xy_graph_l96(self, Xn, n_step, file_path):
        logger.info('xy_graph_l96()')
        fig = plt.figure()
        self.make_file(file_path)

        x_length = np.arange(0, len(Xn), step=1)
        for i in range(0, n_step):
            if i%10 == 0:
                plt.plot(x_length, Xn[:, i])
                plt.grid(color='k', linestyle='dotted', linewidth=0.5)
                file_name =  file_path + str(i) + ".png"
                plt.savefig(file_name)
                plt.close()

    def xy_graph_hovmoller(self, Xn_list, file_path):
        logger.info('xy_graph_l96()')
        
        self.make_file(file_path)
        F_list = [0.1, 0.5, 1.0, 3.0, 5.0, 8.0] # task1-3専用

        for i, val in enumerate(F_list):
            fig, ax = plt.subplots(1, 1)
            plt.xlim(0, 40)
            image1 = ax.contourf(Xn_list[i][:, :].T, cmap='gist_rainbow_r',  levels=10)
            cb1 = fig.colorbar(image1, ax=ax,  orientation='horizontal')
            file_name =  file_path + str(val) + ".png"
            plt.savefig(file_name)
            plt.close()

    def make_file(self, file_path):
        logger.info('make_saveFile()')
        try:
            os.mkdir(file_path)
        except FileExistsError:
            pass