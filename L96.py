import numpy as np
from numba import njit


###############################################################
# Lorentz 96 model
###############################################################
@njit
def l96(x_in, force, n):
    dx_dt = np.zeros_like(x_in)
    dx_dt[0] = (x_in[1] - x_in[n-2]) * x_in[n-1] - x_in[0] + force
    dx_dt[n-1] = (x_in[0] - x_in[n-3]) * x_in[n-2] - x_in[n-1] + force
    for i in range(1, n-1):
        dx_dt[i] = (x_in[i+1] - x_in[i-2]) * x_in[i-1] - x_in[i] + force
    return dx_dt


###############################################################
# solve Lorentz96 model with Runge-Kutta method (4-th order)
###############################################################
@njit
def rk4(x_in, force, n, dt):
    k1_rk4 = l96(x_in,              force, n) * dt
    k2_rk4 = l96(x_in + k1_rk4*0.5, force, n) * dt
    k3_rk4 = l96(x_in + k2_rk4*0.5, force, n) * dt
    k4_rk4 = l96(x_in + k3_rk4,     force, n) * dt
    return x_in + (k1_rk4 + 2*k2_rk4 + 2*k3_rk4 + k4_rk4) / 6


###############################################################
# solve Lorentz96 model with Runge-Kutta method (second order)
###############################################################
@njit
def rk2(x_in, force, n, dt):
    k1_rk2 = l96(x_in, force, n)
    k2_rk2 = l96(x_in + k1_rk2*dt*0.5, force, n)
    return x_in + k2_rk2*dt


###############################################################
# solve Lorentz96 model with Euler method
###############################################################
@njit
def euler(x_in, force, n, dt):
    return x_in + l96(x_in, force, n) * dt


###############################################################
# Iteration
###############################################################
@njit
def run(x, iteration, x_out, force, n, dt):
    x_out[0, :] = x
    for i_run in range(1, iteration):
        x = rk4(x, force, n, dt)
        x_out[i_run, :] = x
    return x_out


@njit
def run_rk2(x, iteration, x_out, force, n, dt):
    x_out[0, :] = x
    for i_run in range(1, iteration):
        x = rk2(x, force, n, dt)
        x_out[i_run] = x
    return x_out


@njit
def run_euler(x, iteration, x_out, force, n, dt):
    x_out[0, :] = x
    for i_run in range(1, iteration):
        x = euler(x, force, n, dt)
        x_out[i_run] = x
    return x_out


@njit
def run_model(x_in, force, n, dt):
    return rk4(x_in, force, n, dt)


###############################################################
# function to calculate RMSE
###############################################################
@njit
def rmse(x1, x2):
    return np.sqrt(np.sum(np.square(x1 - x2))/float(len(x1)))

