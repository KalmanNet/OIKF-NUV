import numpy as np

def MSE(time, GT, y_noisy, y_noisy_with_outliers, x_hat_noisy, x_hat_noisy_with_outliers, len_r_vec):
  
  def inner_calculation(GT, ref):

    MSE_i     = np.mean(np.mean((GT - ref ) ** 2,axis=0),axis=0)
    MSE_lin   = np.mean(MSE_i)
    MSE_dB    = 10 * np.log10(MSE_lin)
    sigma_lin = np.std(MSE_i)
    sigma_dB  = 10 * np.log10(MSE_lin + sigma_lin) - MSE_dB
 
    return MSE_dB, sigma_dB

  P_vec = np.concatenate((y_noisy, x_hat_noisy, y_noisy_with_outliers, x_hat_noisy_with_outliers), axis=0)
  len_Pvec = P_vec.shape[0]//2
  P_MSE_dB   = np.zeros((len_Pvec, len_r_vec))
  P_sigma_dB = np.zeros((len_Pvec, len_r_vec))
  
  for k in range(len_r_vec):
    for j in range(len_Pvec):
       P_MSE_dB[j,k], P_sigma_dB[j,k] = inner_calculation(GT[:,:,:], P_vec[2*j:2*j+2, :, k, :])
   
  return P_MSE_dB, P_sigma_dB
