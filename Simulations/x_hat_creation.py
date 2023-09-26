from KF_algorithms import *
import numpy as np

def x_hat_create(data, r_squared_vec,  del_t, time_t, scl, p):

  N = data.shape[4]
  len_r_vec = data.shape[3]

  x_hat_noisy =np.zeros((10, len(time_t), len_r_vec, N))
  x_hat_noisy_with_outliers = np.zeros((10, len(time_t), len_r_vec, N))

  for j in range(N):
    print(j)
    for k in range(len_r_vec):

      F = np.array([[1,del_t],[0,1]])
      H = np.array([[1,0],[0,1]])
      P = np.diag((1,1))
      I = np.eye(2)

      r_squared = np.array((1, 1)) * r_squared_vec[k]
      q_squared = np.array((0.1,0.1))
      R = np.diag(r_squared)
      r_squared = np.expand_dims(r_squared, axis=1)
      q_squared = np.expand_dims(q_squared, axis=1)
      Q = np.array([[(1/3)*(del_t**3),(1/2)*(del_t**2)],
                  [(1/2)*(del_t**2),del_t]]) * q_squared

      x_init = np.array([[0], [0]]) 

      x_hat_noisy[0:2,:,k,j]                        = KF(data[:,0,:,k,j], x_init, P, F, H, R, Q, I)
      x_hat_noisy_with_outliers[0:2,:,k,j]          = KF(data[:,1,:,k,j], x_init, P, F, H, R, Q, I)
      
      x_hat_noisy[2:4,:,k,j]                       = WRKF(data[:,0,:,k,j], x_init, P, F, H, R, Q, I)   
      x_hat_noisy_with_outliers[2:4,:,k,j]         = WRKF(data[:,1,:,k,j], x_init, P, F, H, R, Q, I)      
 
      x_hat_noisy[4:6,:,k,j]                 = OIKF_AM(data[:,0,:,k,j], x_init, P, F, H, R, Q, I, r_squared)
      x_hat_noisy_with_outliers[4:6,:,k,j]  = OIKF_AM(data[:,1,:,k,j], x_init, P, F, H, R, Q, I, r_squared)

      x_hat_noisy[6:8,:,k,j]                 = OIKF_EM(data[:,0,:,k,j], x_init, P, F, H, R, Q, I, r_squared)
      x_hat_noisy_with_outliers[6:8,:,k,j]       = OIKF_EM(data[:,1,:,k,j], x_init, P, F, H, R, Q, I, r_squared)

      x_hat_noisy[8:10,:,k,j]                = Chi_squared(data[:,0,:,k,j], x_init, P, F, H, R, Q, I)
      x_hat_noisy_with_outliers[8:10,:,k,j]  = Chi_squared(data[:,1,:,k,j], x_init, P, F, H, R, Q, I)
  
  np.save('results/x_hat_noisy_p='+str(p)+'_scl='+str(scl), x_hat_noisy)
  np.save('results/x_hat_noisy_with_outliers_p='+str(p)+'_scl='+str(scl), x_hat_noisy_with_outliers)

  return  x_hat_noisy, x_hat_noisy_with_outliers
