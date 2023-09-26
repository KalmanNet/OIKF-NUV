import numpy as np
from numpy.linalg import inv
from scipy.stats import chi2, norm


"""Kalman Filter"""
def KF(data, time_t, len, r_squared, q_squared, x_init, P_init):
    
    x_KF = np.zeros((2, 1, len))
    P_KF = np.zeros((2, 2, len))
    
    x_KF[:, :, 0] = x_init
    P_KF[:, :, 0] = P_init

    for i in range(1, len):

      del_t = (time_t[i] - time_t[i - 1])
      F = np.array([[1, del_t], [0, 1]])
      H = np.array([[1, 0]])
      I = np.eye(2)
      R = np.array((r_squared))
      Q = np.array([[(1 / 3) * (del_t ** 3), (1 / 2) * (del_t ** 2)],
                  [(1 / 2) * (del_t ** 2), del_t]]) * q_squared
      y = data[i]
     
      # Predict
      x_KF[:, :, i] = F @ x_KF[:, :, i - 1]
      P_KF[:, :, i] = F @ P_KF[:, :, i - 1] @ F.transpose() + Q
     
      # Update
      dy = y - H @ x_KF[:, :, i]
      S = H @ P_KF[:, :, i] @ H.transpose() + R
      K = P_KF[:, :, i] @ H.transpose() / S
      x_KF[:, :, i]  =  x_KF[:, :, i]  + K @ dy
      P_KF[:, :, i] = (I - K @ H) @ P_KF[:, :, i]

    return x_KF[:, 0, :]

"""Chi_squared"""
def Chi_squared(data, time_t, len, r_squared, q_squared, x_init, P_init):
   
    order = 1
    Chi_square_test = chi2.isf(0.05, order) 
   
    x_Chi = np.zeros((2, 1, len))
    P_Chi = np.zeros((2, 2, len))

    x_Chi[:, :, 0] = x_init
    P_Chi[:, :, 0] = P_init

    for i in range(1, len):

      del_t = (time_t[i] - time_t[i - 1])
      F = np.array([[1, del_t], [0, 1]])
      H = np.array([[1, 0]])
      I = np.eye(2)
      R = np.array((r_squared))
      Q = np.array([[(1 / 3) * (del_t ** 3), (1 / 2) * (del_t ** 2)],
                  [(1 / 2) * (del_t ** 2), del_t]]) * q_squared
      y = data[i]
     
      # Predict
      x_Chi[:, :, i] = F @ x_Chi[:, :, i - 1]
      P_Chi[:, :, i] = F @ P_Chi[:, :, i - 1]  @ F.transpose() + Q
      dy = y - H @  x_Chi[:,:,i]
      S = H @ P_Chi[:, :, i] @ H.transpose() + R

      Chi_square = dy @ inv(S) @ dy.transpose()
      if Chi_square < Chi_square_test:
        # Update
        K = P_Chi[:,:,i] @ H.transpose() / S
        x_Chi[:, :, i]  =  x_Chi[:, :, i]  + K @ dy
        P_Chi[:, :, i] = (I - K @ H) @ P_Chi[:, :, i]

    return x_Chi[:, 0, :]
  
"""ORKF"""
def ORKF(data, time_t, len, r_squared, q_squared, x_init, P_init):
    
    x_ORKF = np.zeros((2, 1, len))
    P_ORKF = np.zeros((2, 2, len))
    
    x_ORKF[:, :, 0] = x_init
    P_ORKF[:, :, 0] = P_init
   
    N = 60
    s = 3
    
    for i in range(1, len):

      del_t = (time_t[i] - time_t[i - 1])
      F = np.array([[1, del_t], [0, 1]])
      H = np.array([[1, 0]])
      I = np.eye(2)
      R = np.array((r_squared))
      Q = np.array([[(1 / 3) * (del_t ** 3), (1 / 2) * (del_t ** 2)],
                  [(1 / 2) * (del_t ** 2), del_t]]) * q_squared
      y = data[i]
      
      # Predict
      x_ORKF[:, :, i] = F @ x_ORKF[:, :, i-1]
      P_ORKF[:, :, i] = F @ P_ORKF[:, :, i-1] @ F.transpose() + Q
      GAMMA = R
      dy = y - H @  x_ORKF[:, :, i]

      for n in range(N - 1):
        S = H @ P_ORKF[:, :, i] @ H.transpose() + GAMMA
        K = P_ORKF[:, :, i] @ H.transpose() / S
        x_ORKF[:, :, i] = x_ORKF[:, :, i]  + K @ dy
        P_ORKF[:, :, i] = (I - K @ H) @ P_ORKF[:, :, i] @ (I - H.transpose() @ K.transpose()) + K * GAMMA * K.transpose()
        dy = y - H @ x_ORKF[:, :, i]
        GAMMA = (s * R + dy ** 2 + H @ P_ORKF[:, :, i] @ H.transpose()) / (s + 1)
    
    return x_ORKF[:,0,:]

"""OIKF_AM"""
def OIKF_AM(data, time_t, len, r_squared, q_squared, x_init, P_init):
    
    x_AM = np.zeros((2, 1, len))
    P_AM = np.zeros((2, 2, len))

    x_AM[:, :, 0] = x_init
    P_AM[:, :, 0] = P_init

    for i in range(1, len):
   
      del_t = (time_t[i] - time_t[i - 1])
      F = np.array([[1, del_t], [0, 1]])
      H = np.array([[1, 0]])
      I = np.eye(2)
      Q = np.array([[(1 / 3) * (del_t ** 3), (1 / 2) * (del_t ** 2)],
                    [(1 / 2) * (del_t ** 2), del_t]]) * q_squared
      y = data[i]

      # Predict
      x_pred = F @ x_AM[:, :, i-1]
      P_pred = F @ P_AM[:, :, i-1] @ F.transpose() + Q
      dy = y - H @ x_pred

      gamma_pos = np.maximum((dy ** 2) - r_squared, 0)
      # gamma_pos = 0

      for j in range(0, 10):
        # Update
        R = np.array((r_squared + gamma_pos))
        S = H @ P_pred @ H.transpose() + R
        K = P_pred @ H.transpose() / S
        x_AM[:, :, i] = x_pred + K @ dy
        P_AM[:, :, i] = (I - K @ H) @ P_pred
        
        # Outlier variance estimate
        v = y - H @ x_AM[:, :, i]
        gamma_pos = np.maximum(v ** 2 - r_squared, 0)

        # if gamma_pos < 20:
        #     gamma_pos = 0

    return  x_AM[:, 0, :]

"""OIKF_EM"""
def OIKF_EM(data, time_t, len, r_squared, q_squared, x_init, P_init):

    x_EM = np.zeros((2, 1, len))
    P_EM = np.zeros((2, 2, len))
    
    x_EM[:, :, 0] = x_init
    P_EM[:, :, 0] = P_init

    for i in range(1, len):
    
      del_t = (time_t[i] - time_t[i - 1])
      F = np.array([[1, del_t], [0, 1]])
      H = np.array([[1, 0]])
      I = np.eye(2)
      Q = np.array([[(1 / 3) * (del_t ** 3), (1 / 2) * (del_t ** 2)],
                    [(1 / 2) * (del_t ** 2), del_t]]) * q_squared
      y = data[i]
 
      # Predict
      x_pred = F @ x_EM[:, :, i - 1]
      P_pred = F @ P_EM[:, :, i - 1] @ F.transpose() + Q
      dy = y - H @ x_pred
 
      # Outlier variance estimate
      moment_II = (P_pred + x_pred @ x_pred.transpose())
      HPH_pos = H @ moment_II @ H.transpose()
      gamma_pos = np.maximum(y ** 2 - r_squared - 2 * y * (H @ x_pred) + HPH_pos, 0)

      for j in range(0, 10):
        
        # Update
        R = np.array((r_squared + gamma_pos))
        S = H @ P_pred @ H.transpose() + R
        K = P_pred @ H.transpose() / S
        x_EM[:, :, i]  =  x_pred + K @ dy
        P_EM[:, :, i] = (I - K @ H) @ P_pred

        moment_II = (P_EM[:, :, i] + x_EM[:, :, i] @ x_EM[:, :, i].transpose())
        HPH_pos = H @ moment_II @ H.transpose()
        gamma_pos = np.maximum(y ** 2 - r_squared - 2 * y * (H @ x_EM[:, :, i]) + HPH_pos, 0)

        # if gamma_pos<4:
        #     gamma_pos=0

    return  x_EM[:,0,:]
