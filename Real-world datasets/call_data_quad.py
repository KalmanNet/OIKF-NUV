import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import bernoulli,rayleigh
import os

def call_data_quad(direc):

    if direc=="quad_hor":
        data = np.loadtxt('Quadrotor/Periodic Motion/Horizontal/GT_3.csv', delimiter = ",", skiprows=1)
        data = data[250:1430]

    elif direc=="quad_ver":
        data = np.loadtxt('Quadrotor/Periodic Motion/Vertical/GT_4.csv', delimiter = ",", skiprows=1)
        data = data [320:1360]

    data_GT = -data[:,3]
    time = (data[:,0] - data[0,0])
    N = len(time)

    no_outliers = 3
    scl = 20

    continous_outlier_index = np.random.randint(0, len(time), no_outliers)
    continous_outlier_intens = rayleigh.rvs(size=no_outliers, scale=scl)

    data_GPS_outlier = np.copy(data_GT)

    for i in range(30):
        data_GPS_outlier[continous_outlier_index+i] =  data_GPS_outlier[continous_outlier_index+i] + continous_outlier_intens

    if (os.path.isfile('Quadrotor/data_'+direc+'.npy') == True):
        data_GPS_outlier = np.load('Quadrotor/data_'+direc+'.npy')
    else:
        np.save('Quadrotor/data_'+direc+'.npy', data_GPS_outlier)

    plt.plot(time, data_GPS_outlier, label="GPS")
    plt.plot(time, data_GT, label="GT")

    plt.xlabel('Time [s]', size='20')
    plt.ylabel('Position [m]', size='20')
    plt.grid()
    plt.legend(fontsize=14)
    plt.show()

    return data_GT, data_GPS_outlier, time
