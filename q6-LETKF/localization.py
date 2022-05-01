# EXTERNAL LIBRARIES
from logging import getLogger, DEBUG, basicConfig
from matplotlib.pyplot import step
import numpy as np

# INTERNAL LIBRARIES
from plot import Plot_Methods

# PARAMETER
N = 40
L = np.zeros((N, N))
sigmas = [1, 3, 5]

class Localization:
    # METHOD
    def get_distance(self, i, j):
        normal_data = abs(i-j)
        reverse_data = 40-abs(i-j)
        return min(normal_data, reverse_data)

    def get_L(self, sigma):
        for i in range(N):
            for j in range(N):
                d = self.get_distance(i, j)
                if 2*np.sqrt(10.0/(3.0*sigma)):
                    L[i, j] = np.exp(-(d*d)/(2.0*sigma))
                else:
                    L[i, j] = 0.0
        return L