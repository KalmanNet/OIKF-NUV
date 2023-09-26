import numpy as np

def MSE(time, GT, y, x_hat, q_squared_vec, r_squared_vec):

  def inner_calculation(GT, ref):
    MSE_i     = (GT - ref) ** 2
    MSE_lin   = np.mean(MSE_i)
    MSE_dB    = 10 * np.log10(MSE_lin)
    MSE_m = np.sqrt(MSE_lin)
    return MSE_dB, MSE_m
  
  len_r_vec = len(r_squared_vec)
  len_q_vec = len(q_squared_vec)

  len_x_hat  = x_hat.shape[0]
  P_MSE_dB   = np.zeros((1 + len_x_hat, len_r_vec, len_q_vec))
  P_MSE_m = np.zeros((1 + len_x_hat, len_r_vec, len_q_vec))

  for i in range(len_r_vec):
    for k in range(len_q_vec):
      for j in range(len_x_hat):
        P_MSE_dB[j, i ,k], P_MSE_m[j, i, k] = inner_calculation(GT, x_hat[j, 0, :, i, k])
      P_MSE_dB[len_x_hat, i, k], P_MSE_m[len_x_hat, i, k] = inner_calculation(GT, y)

  return P_MSE_dB, P_MSE_m
