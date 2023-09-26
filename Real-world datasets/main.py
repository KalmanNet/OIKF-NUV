from call_data_segway import *
from call_data_quad import *
from create_data import *
from matrices_calculation import *
from analysis_and_results import *

import numpy as np
import os.path

def main():

    # # Segway Robot # #
    vehicular = 'Segway'

    data_GT_x, data_GT_y, data_GPS_x, data_GPS_y, time_GPS, time_GT = call_data_segway()

    #North direction
    data_y = data_GPS_x
    data_GT = data_GT_x
    direc = 'north'
    #
    # # # East direction
    # data_y = data_GPS_y
    # data_GT = data_GT_y
    # direc = 'east'


    # # Quadrotor # #
    # vehicular = 'Quadrotor'
    # direc = 'quad_hor' #'quad_ver'
    # data_GT, data_GPS, time_GPS = call_data_quad(direc)
    # data_y = data_GPS
    # data_GT = data_GT
    #

    file_name_MSE_dB = vehicular +'/results/P_MSE_dB_'+direc+'.npy'
    file_name_MSE_m = vehicular +'/results/P_MSE_m_'+direc+'.npy'
    file_name_x_hat = vehicular +'/results/x_hat_'+direc+'.npy'

    q_squared_vec = np.array((1, 1e-1, 1e-2, 1e-3, 1e-4))
    r_squared_vec = np.array((1, 1e1, 1e2, 1e3, 1e4))


    if (os.path.isfile(file_name_x_hat) == True):
      x_hat = np.load(file_name_x_hat)

    else:
      x_hat = Create_data(data_y, time_GPS, r_squared_vec, q_squared_vec)
      np.save(file_name_x_hat, x_hat)


    P_MSE_dB, P_MSE_m = MSE(time_GPS, data_GT, data_y, x_hat, q_squared_vec, r_squared_vec)
    np.save(file_name_MSE_dB, P_MSE_dB)
    np.save(file_name_MSE_m, P_MSE_m)

    min_vec = np.zeros(x_hat.shape[0] + 1)
    index_vec = np.zeros((x_hat.shape[0] + 1, 2))

    for i in range(x_hat.shape[0]):
      min_vec[i] = P_MSE_m[i, :, :].min()
      index = np.where(P_MSE_m[i, :] == min_vec[i])
      index_vec[i, :] = np.array(index)[:, 0]

    plot_MSE(data_y,data_GT, x_hat, time_GPS, P_MSE_dB, P_MSE_m, index_vec, vehicular, direc)

 
if __name__ == "__main__":
    main()
