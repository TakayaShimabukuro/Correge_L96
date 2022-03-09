import matplotlib.pyplot as plt
from logging import getLogger
import os
logger = getLogger(__name__)



class Plot_Methods:
    def xy_graph_l96(self, model):
        logger.info('xy_graph_l96()')
        fig = plt.figure()
        self.make_saveFile()

        for i in range(0, model.n_step):
            if i%10 == 0:
                plt.plot(model.Xn[:, i])
                plt.xlabel("time(days)")
                plt.ylabel("RMSE")
                #plt.xticks(np.arange(0, 110, step=10))
                plt.grid(color='k', linestyle='dotted', linewidth=0.5)
                name =  "./q2/result/" + str(i) + ".png"
                plt.savefig(name)
                plt.close()
    

    def make_saveFile(self):
        logger.info('make_saveFile()')
        try:
            new_dir_path = './q2/result'
            os.mkdir(new_dir_path)
        except FileExistsError:
            pass