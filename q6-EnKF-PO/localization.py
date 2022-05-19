# EXTERNAL LIBRARIES
import numpy as np

class Localization:
    #PARAMETER
    def __init__(self, N):
        self.N = N
        self.L = np.zeros((N, N))
        self.sigmas =  [1, 3, 5]


    # METHOD
    def get_distance(self, i, j):
        normal_data = abs(i-j)
        reverse_data = 40-abs(i-j)
        return min(normal_data, reverse_data)

    def get_L(self, sigma):
        for i in range(self.N):
            for j in range(self.N):
                d = self.get_distance(i, j)
                if 2*np.sqrt(10.0/(3.0*sigma)):
                    self.L[i, j] = np.exp(-(d*d)/(2.0*sigma))
                else:
                    self.L[i, j] = 0.0
        return self.L