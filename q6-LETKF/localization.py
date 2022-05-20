# EXTERNAL LIBRARIES
import numpy as np

class Localization:
    #PARAMETER
    def __init__(self, N):
        self.N = N
        self.L = np.zeros((N, N))

    # METHOD
    def get_distance(self, i, j):
        normal_data = abs(i-j)
        reverse_data = self.N-abs(i-j)
        return min(normal_data, reverse_data)

    def get_L(self, sigma):
        for i in range(self.N):
            for j in range(self.N):
                d = self.get_distance(i, j)
                if d < 2 * np.sqrt(10/3) * sigma:
                    self.L[i, j] = np.exp(-d * d * 0.5 / (sigma * sigma))
        return self.L