from x_hat_creation import *
from analysis_and_results import *
from data_generation import *
from matrices_calculation import *

import numpy as np
import os.path

def main():
   
  r_squared_vec = np.array((10, 1, 0.1, 0.01, 0.001))
  r_squared_vec_dB = -10 * np.log10(r_squared_vec)
  
  p = 0.2
  scl = 50

  del_t = 1e-1
  end_time = 10
  time_t = np.linspace(0, end_time,  int((end_time/del_t))+1)

  data_GT_file = 'results/data_GT.npy'
  data_y_file = 'results/data_y_p='+str(p)+'_scl='+str(scl)+'.npy'
  if not os.path.isfile(data_GT_file) or not os.path.isfile(data_y_file):
    Data_Generation(p, scl)

  data_GT = np.load(data_GT_file)
  data_y = np.load(data_y_file)

  N = data_GT.shape[2]
    
  if (os.path.isfile('results/x_hat_noisy_p='+str(p)+'_scl='+str(scl)+'.npy') and os.path.isfile('results/x_hat_noisy_with_outliers_p='+str(p)+'_scl='+str(scl)+'.npy')):
    x_hat_noisy = np.load('results/x_hat_noisy_p='+str(p)+'_scl='+str(scl)+'.npy')
    x_hat_noisy_with_outliers = np.load('results/x_hat_noisy_with_outliers_p='+str(p)+'_scl='+str(scl)+'.npy')

  else:

    x_hat_noisy, x_hat_noisy_with_outliers = x_hat_create(data_y[:,:,:,:,0:N], r_squared_vec,del_t, time_t, scl, p)
 
  len_r_vec = len(r_squared_vec)

  P_MSE_dB, P_sigma_dB = MSE(time_t, data_GT[:,:,0:N], data_y[:,0,:,:,0:N], data_y[:,1,:,:,0:N],
                              x_hat_noisy[:,:,:,0:N], x_hat_noisy_with_outliers[:,:,:,0:N], len_r_vec)
 
  plot_MSE(r_squared_vec_dB, P_MSE_dB, P_sigma_dB, scl, p)

     
if __name__ == "__main__":
    main()
 
