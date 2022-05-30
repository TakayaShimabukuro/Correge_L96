# EXTERNAL LIBRARIES
import numpy as np

class Localization:
    #PARAMETER
    def __init__(self, N):
        self.N = N

    # METHOD
    def get_distance(self, i, j):
        normal_data = abs(i-j)
        reverse_data = self.N-abs(i-j)
        return min(normal_data, reverse_data)

    def get_L(self, sigma, H, target):
        n_sensors = H.shape[0]
        n_grids = H.shape[1]
        localization_func = np.zeros(n_sensors)

        # calculate localization function
        for i in range(n_grids):
            if np.any(H[:, i] == 1):
                distance = np.abs(i - target)
                if distance > self.N*0.5:
                    distance = self.N - distance
                if distance < 2 * np.sqrt(10/3) * sigma:
                    obs_loc = np.argmax(H[:, i])
                    localization_func[obs_loc] = np.exp(-distance * distance * 0.5 / (sigma * sigma))
        return localization_func