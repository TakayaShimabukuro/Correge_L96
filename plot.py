import matplotlib.pyplot as plt
from logging import getLogger
import os
import numpy as np
logger = getLogger(__name__)



class Plot_Methods:
    def xy_graph_l96(self, model, file_path):
        logger.info('xy_graph_l96()')
        fig = plt.figure()
        self.make_saveFile(file_path)

        x_length = np.arange(0, len(model.Xn), step=1)
        for i in range(0, model.n_step):
            if i%10 == 0:
                plt.plot(x_length, model.Xn[:, i])
                plt.grid(color='k', linestyle='dotted', linewidth=0.5)
                file_name =  file_path + str(i) + ".png"
                plt.savefig(file_name)
                plt.close()
    

    def make_saveFile(self, file_path):
        logger.info('make_saveFile()')
        try:
            os.mkdir(file_path)
        except FileExistsError:
            pass