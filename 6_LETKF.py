import numpy as np
from matplotlib import pyplot as plt
from numba import njit, f8
import sys

from L96 import run, run_model, rmse
from logging import getLogger, DEBUG, basicConfig
# DEBUG SETTING
logger = getLogger(__name__)
logger.setLevel(DEBUG)
basicConfig(filename='console.log', level=DEBUG, filemode='w')

###############################################################
# To check elapsed time
###############################################################
import time
from contextlib import contextmanager


plt.rcParams["font.size"] = 12


@contextmanager
def timer():
    t = time.perf_counter()
    yield None
    print('Elapsed:', time.perf_counter() - t)


###############################################################
# parameters for Lorentz96
###############################################################
# Forcing
force = 8.0

# Number of division
n = 40

# number of observation
n_obs = 20

###############################################################
# parameters for Ensemble Kalman Filter
###############################################################
# ensemble size
m_ensemble_size = 8

# Localization length scale
localization_length_scale = 3

# inflation
inflation = 1.03

###############################################################
# calculation settings
###############################################################
# number of days for main run
days_main = 365

# interval to record true values [hours]
rec_interval = 1

# time step (dt=0.2 == 1 day)
dt = 0.2 / (4 * rec_interval)

# number of iteration for 1 day
iter_day = int(0.2/dt+0.1)

# number of iteration for main calculation
main_iter = int(iter_day * days_main + 0.001)

# number of iteration for main calculation
rec_iter = int(main_iter / rec_interval + 0.001)


###############################################################
# Data Assimilation
###############################################################
@njit
def define_h_and_r(p, n):
    r_obs_cov = np.identity(p)

    h_obs_operator = np.zeros((p, n))

    remainder = n % p
    obs_interval = int(n/p+1e-9)
    obs_loc = 0
    for i_obs in range(p-remainder):
        h_obs_operator[i_obs, obs_loc] = 1
        obs_loc += obs_interval
    if remainder > 0:
        for i_obs in range(p-remainder, p):
            h_obs_operator[i_obs, obs_loc] = 1
            obs_loc += obs_interval + 1
    return h_obs_operator, r_obs_cov


@njit
def obs_operator(h_obs_operator, xb):
    return h_obs_operator @ xb


@njit
def gauss_func(target_grid, sigma, h_obs_operator):
    n_sensors = h_obs_operator.shape[0]
    n_grids = h_obs_operator.shape[1]
    localization_func = np.zeros(n_sensors)

    # calculate localization function
    for i in range(n_grids):
        if np.any(h_obs_operator[:, i] == 1):
            distance = np.abs(i - target_grid)
            if distance > n*0.5:
                distance = n - distance
            if distance < 2 * np.sqrt(10/3) * sigma:
                obs_loc = np.argmax(h_obs_operator[:, i])
                localization_func[obs_loc] = np.exp(-distance * distance * 0.5 / (sigma * sigma))
    return localization_func


@njit
def forecast_enkf(xb, xa, ensemble_size):
    for j in range(ensemble_size):
        xb[:, j] = run_model(xa[:, j], force, n, dt)
    return xb


@njit
def cal_zb(zb, xb_mean, ensemble_size):
    for j in range(ensemble_size):
        zb[:, j] -= xb_mean
    return zb / np.sqrt(ensemble_size - 1)


@njit
def analysis_letkf(xb, y_o, h_obs_operator, ensemble_size, sigma, delta):
    xb_mean = np.sum(xb, axis=1) / ensemble_size
    xb_mean_ens = (np.ones((ensemble_size, 1)) @ xb_mean.reshape(1, -1)).T

    # preparation step for grid loop
    zb = cal_zb(xb.copy(), xb_mean, ensemble_size) * delta
    # zb = cal_zb(xb.copy(), xb_mean, ensemble_size) * np.sqrt(delta)
    yb = h_obs_operator @ zb
    print("---")
    print(h_obs_operator.shape)
    print(zb.shape)
    d_ob = y_o - obs_operator(h_obs_operator, xb_mean)

    xa = np.zeros_like(xb)

    # analysis step
    for i in range(n):
        # R-localization
        if sigma > 0:
            localization_func = gauss_func(i, sigma, h_obs_operator)
        else:
            localization_func = np.ones(n_obs)
        r_local_inv = np.diag(np.ones(n_obs) * localization_func)
        
        # eigen value decomposition
        pa_tilde_inv = np.identity(ensemble_size) + yb.T @ r_local_inv @ yb
        eig_val, eig_vec = np.linalg.eigh(pa_tilde_inv)
        pa_tilde = eig_vec @ np.diag(1/eig_val) @ eig_vec.T
        pa_tilde_sqrt = eig_vec @ np.diag(1/np.sqrt(eig_val)) @ eig_vec.T

        # transpose matrix
        trans_mat = (pa_tilde @ yb.T @ r_local_inv @ d_ob).reshape(-1, 1) @ np.ones((1, ensemble_size)) + \
                    pa_tilde_sqrt * np.sqrt(ensemble_size - 1)
        xa[i, :] = (xb_mean_ens + zb @ trans_mat)[i, :]

    return xa


@njit
def letkf_main_loop(rec_iter, xa, ensemble_size, x_obs, x_true, h_obs_operator, sigma, alpha):
    rmse_rec = np.zeros(rec_iter)
    trace_rec = np.zeros(rec_iter)

    xb = np.zeros_like(xa)

    for iter in range(rec_iter):
        # forecast step
        xb = forecast_enkf(xb, xa, ensemble_size)

        y_o = h_obs_operator @ x_obs[iter, :]

        xa = analysis_letkf(xb, y_o, h_obs_operator, ensemble_size, sigma, alpha)

        xa_mean = np.sum(xa, axis=1) / ensemble_size

        delta_xa = np.zeros_like(xa)
        for i in range(ensemble_size):
            delta_xa[:, i] = xa[:, i] - xa_mean
        pa = delta_xa @ delta_xa.T / (ensemble_size - 1)

        rmse_rec[iter] = rmse(xa_mean, x_true[iter, :])
        trace_rec[iter] = np.sqrt(np.trace(pa) / float(n))

        # plt.figure()
        # plt.imshow(trans_mat)
        # plt.colorbar()
        # plt.show()

        # plt.figure()
        # plt.plot(np.sum(trans_mat, axis=0))
        # # plt.plot(np.sum(trans_mat, axis=1))
        # plt.show()

        # for i in range(ensemble_size):
        #     plt.plot(xa[:, i], c='blue', lw=0.2)
        # plt.plot(xa_mean, c='blue', lw=2)
        # plt.plot(x_true[iter, :], c='red', lw=2)
        # plt.plot(x_obs[iter, :], c='green', lw=2)
        # plt.show()

    return rmse_rec, trace_rec


# @njit
def exe_letkf(x_obs, x_true, ensemble_size, sigma, alpha):

    # define parameters for Kalman Filter
    h_obs_operator, r_obs_cov = define_h_and_r(n_obs, n)

    # initial_value
    xa = np.ones((ensemble_size, 1)) @ x_true[100, :].reshape(1, -1)
    xa = xa.T
    np.random.seed(113)
    xa += np.random.normal(size=xa.shape) * 5


    rmse_rec, trace_rec = letkf_main_loop(rec_iter, xa, ensemble_size, x_obs, x_true, h_obs_operator, sigma, alpha)
    return rmse_rec, trace_rec


def cal_obs_rmse(x_obs, x_true):
    rmse_obs = np.zeros(rec_iter)
    for iter in range(rec_iter):
        rmse_obs[iter] = rmse(x_obs[iter, :], x_true[iter, :])
    return rmse_obs


def read_files():
    f_obs = './output/obs.csv'
    f_true = './output/true.csv'
    obs_read = np.loadtxt(f_obs, delimiter=',')
    true_read = np.loadtxt(f_true, delimiter=',')
    return obs_read, true_read


#####################################################################
# Main
#####################################################################
# with timer():
#     # read obs and true data
#     obs, true = read_files()
#
#     rmse_out, trace_out = exe_letkf(obs, true, m_ensemble_size, localization_length_scale, inflation)
#
#     print(np.mean(rmse_out[50:]))
#
# xticks_day = np.linspace(0, days_main, main_iter)
#
# plt.figure(figsize=(12, 4))
# plt.title(r'L96 + LETKF (m=%s, $\sigma$=%s, $\alpha$=%s)' % (m_ensemble_size, localization_length_scale,
#                                                               int((inflation-0.999)*100)))
# plt.plot(xticks_day, rmse_out, c='black', label='RMSE m=%s' % m_ensemble_size)
# plt.plot(xticks_day, trace_out, c='grey', label='Spread')
# plt.xlim(0, days_main)
# plt.ylim(0, 1)
# plt.xlabel('day')
# plt.ylabel('RMSE')
# plt.legend()
# plt.grid()
# plt.savefig('./figs/6_LETKF/6_RMSE_LETKF_m-%s_sigma%s_alpha%s.png' % (m_ensemble_size, localization_length_scale,
#                                                                       int((inflation-0.999)*100)))
# plt.show()
# plt.close()


#####################################################################
# Make colormap
#####################################################################
n_exp = 11
sigma_list = np.arange(1, n_exp)
inflation_list = 1 + np.arange(1, n_exp) * 0.01
rmse_ave = np.zeros((n_exp-1, n_exp-1))

for i_loc, localization_length_scale in enumerate(sigma_list):
    for i_inf, inflation in enumerate(inflation_list):
        with timer():
            # read obs and true data
            obs, true = read_files()

            rmse_out, trace_out = exe_letkf(obs, true, m_ensemble_size, localization_length_scale, inflation)

            rmse_ave[i_inf, i_loc] = np.mean(rmse_out[50:])
            print(i_loc, i_inf, np.mean(rmse_out[50:]))

        xticks_day = np.linspace(0, days_main, main_iter)

        plt.figure(figsize=(12, 4))
        plt.title(r'L96 + LETKF (m=%s, $\sigma$=%s, $\delta$=%s)' % (m_ensemble_size, localization_length_scale,
                                                                    int((inflation-0.999)*100)))
        plt.plot(xticks_day, rmse_out, c='black', label='RMSE m=%s' % m_ensemble_size)
        plt.plot(xticks_day, trace_out, c='grey', label='Spread')
        plt.xlim(0, days_main)
        plt.xlabel('day')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid()
        plt.savefig('./figs/6_LETKF/6_RMSE_LETKF_m-%s_sigma%s_delta%s.png' % (m_ensemble_size, localization_length_scale,
                                                                              int((inflation-0.999)*100)))
        # plt.show()
        plt.close()

print('minimum RMSE = ', np.min(rmse_ave))

plt.figure()
cmap = plt.cm.seismic
cmap.set_under('indigo')
cmap.set_over('red')
if m_ensemble_size == 8 and n_obs == 20:
    levels = [0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
else:
    levels = [0.21, 0.225, 0.25, 0.275, 0.3, 0.335, 1]
xx, yy = np.meshgrid(sigma_list, inflation_list)
plt.contourf(xx, yy, rmse_ave, cmap=cmap, levels=levels)
plt.title('RMSE m=%s' % m_ensemble_size)
plt.xlabel('Localization Length Scale ($\sigma$)')
plt.ylabel('Inflation factor ratio($\delta$)')
plt.colorbar()
plt.savefig('./figs/6_LETKF_RMSEmap_m%s_p%s.png' % (m_ensemble_size, n_obs))
plt.show()


# end of file
