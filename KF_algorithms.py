import numpy as np
from numpy.linalg import inv
from scipy.stats import chi2, norm

"""Kalman Filter"""

def KF(data, x_init, P_init, F, H, R, Q, I):
    
    len = data.shape[1]
    x = x_init
    P = P_init

    x_hat = np.zeros((data.shape[0],len))
    x_hat[:,0] = np.squeeze(x_init,axis=1)
    
    for n in range(0,len):

      # prediction
      x = F @ x
      P = F @ P @ F.transpose() + Q
     
      # measurement update
      y = np.expand_dims(data[:,n],axis=1) 
      dy = y - H @ x 
      S = H @ P @ H.transpose() + R
      K = P @ H.transpose() @ inv(S)
      x = x + K @ dy
      P = (I - K @ H) @ P

      x_hat[:,n] = x[:,0]
        
    return x_hat
    
"""Chi-squared """    

def Chi_squared(data, x_init, P_init, F, H, R, Q, I):
   
    len = data.shape[1]
    FA = 0.95
    Chi_square_test = chi2.isf(1-FA, 2)
   
    x = x_init
    P = P_init
    x_hat = np.zeros((data.shape[0],len))
    x_hat[:,0] = np.squeeze(x_init,axis=1)
   
    for n in range(0,len):

      # prediction
      x = F @ x
      P = F @ P @ F.transpose() + Q
      
      # measurement update
      y = np.expand_dims(data[:,n],axis=1) 
      dy = y - H @ x 
      S = H @ P @ H.transpose() + R
      Chi_square = dy.transpose() @ inv(S) @ dy
      if Chi_square < Chi_square_test:
        K = P @ H.transpose() @ inv(S) 
        x = x + K @ dy
        P = (I - K @ H) @ P
        
      x_hat[:,n] = x[:,0]
         
    return x_hat
  
      
"""OIKF-AM """ 
def OIKF_AM(data, x_init, P_init, F, H, R, Q, I, r_squared):

    len = data.shape[1]
    P = np.zeros((data.shape[0],data.shape[0],len))
    x = np.zeros((data.shape[0],1,len))

    P[:,:,0] = P_init
    x[:,:,0] = x_init

    for n in range(0, len):
   
      x_pred = F @ x[:,:,n-1]
      P_pred = F @ P[:,:,n-1]  @ F.transpose() + Q
      y = data[:,n:n+1]
      dy = y - H @ x_pred

      # gamma_pos = np.maximum((dy[0,0]**2) - r_squared[0,0],0)
      # gamma_vel = np.maximum((dy[1,0]**2) - r_squared[1,0],0)
      gamma_pos = 0
      gamma_vel = 0

      for i in range(0,9-1):
        R = np.array([[r_squared[0,0]+gamma_pos,0],[0,r_squared[1,0]+gamma_vel]])
        S = H @ P_pred @ H.transpose() + R
        K = P_pred @ H.transpose() @ inv(S) 
        x[:,:,n] = x_pred + K @ dy
        P[:,:,n] = (I - K @ H) @ P_pred
        
        v = y - H @ x[:,:,n]
        gamma_pos = np.maximum((v[0,0]**2) - r_squared[0,0],0)
        gamma_vel = np.maximum((v[1,0]**2) - r_squared[1,0],0)


    return  x[:,0,:]
  
"""OIKF-EM """

def OIKF_EM(data, x_init, P_init, F, H, R, Q, I, r_squared):
   
    len = data.shape[1]
    P = np.zeros((data.shape[0],data.shape[0],len))
    x = np.zeros((data.shape[0],1,len))
 
    P[:,:,0] = P_init
    x[:,:,0] = x_init

    N = 10

    for n in range(0, len):
      

      x_pred = F @ x[:,:,n-1]
      P_pred = F @ P[:,:,n-1]  @ F.transpose() + Q
      y = data[:,n:n+1]
      dy = y - H @ x_pred
   
      # moment_II = (P_pred + x_pred @ x_pred.transpose())
      # HPH_pos = H[0:1,:] @ moment_II @ H[0:1,:].transpose()
      # HPH_vel = H[1:2,:] @ moment_II @ H[1:2,:].transpose()
      # gamma_pos = np.maximum(y[0,0] ** 2 - r_squared[0,0] - 2 * H[0:1,:] @ (y[0,0] * x_pred) + HPH_pos, 0)[0]
      # gamma_vel = np.maximum(y[1,0] ** 2 - r_squared[1,0] - 2 * H[1:2,:] @ (y[1,0] * x_pred) + HPH_vel, 0)[0]
      gamma_pos = [0]
      gamma_vel = [0]

      for i in range(1,N):
        
        R = np.array([[r_squared[0,0]+gamma_pos[0],0],[0,r_squared[1,0]+gamma_vel[0]]])
        S = H @ P_pred @ H.transpose() + R
        K = P_pred @ H.transpose() @ inv(S) 
        x[:,:,n] = x_pred + K @ dy
        P[:,:,n] = (I - K @ H) @ P_pred
        
        # Calculataion of the outlier variance 
        moment_II = (P[:,:,n] + x[:,:,n] @ x[:,:,n].transpose())
        HPH_pos = H[0:1,:] @ moment_II @ H[0:1,:].transpose()
        HPH_vel = H[1:2,:] @ moment_II @ H[1:2,:].transpose()
        gamma_pos = np.maximum(y[0,0] ** 2 - r_squared[0,0] - 2 * H[0:1,:] @ (y[0,0]* x[:,:,n]) + HPH_pos, 0)[0]
        gamma_vel = np.maximum(y[1,0] ** 2 - r_squared[1,0] - 2 * H[1:2,:] @ (y[1,0]* x[:,:,n]) + HPH_vel, 0)[0]
       
    return  x[:,0,:]


"""WRKF """
def WRKF(data, x_init, P_init, F, H, R, Q, I):
    
    len = data.shape[1]
    x = x_init
    P = P_init
   
    x_hat = np.zeros((data.shape[0],len))
    x_hat[:,0] = np.squeeze(x_init,axis=1)
    N = 30

    for n in range(1, len):
     
      # prediction
      x = F @ x 
      P = Q
      y = data[:,n:n+1]
  
      # measurement update
      for j in range(N):
        dy = y - H @ x
        a = 0.5
        b = 0.5
        w1 = a + 0.5
        w2 = b + (dy.transpose() @ inv(R) @ dy) 
        w = w1/w2
        S = H @ P @ H.transpose() + R/w
        K = P @ H.transpose() @ inv(S) 
        x = x + K @ dy
        P = (I - K @ H) @ P 
        
      x_hat[:,n] = x[:,0]
  
    return x_hat
