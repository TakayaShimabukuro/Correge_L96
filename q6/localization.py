# External Libraries
from logging import getLogger, DEBUG, basicConfig
from matplotlib.pyplot import step
import numpy as np

# Internal Libraries
from plot import Plot_Methods

# PARAMETER
N = 40
path = "./q6/result/"
L = np.zeros((N,N))

for i in range(len(N)):
    for j in range(len(N)):
        d = i-j
        L[i, j] = np.exp()

# INSTANCE
plot = Plot_Methods(path)
plot.Debug(L, "Localization")