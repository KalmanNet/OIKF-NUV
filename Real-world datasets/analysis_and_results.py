import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


def plot_MSE(data_y, data_GT, x_hat, time, P_MSE_dB, P_MSE_m, index_vec, vehicular, direc):

    x_hat_AM = x_hat[3,0,:,int(index_vec[3,0]), int(index_vec[3,1])]

    plt.figure(figsize=(12, 8))

    plt.plot(time / 60, data_y, '.', color='red', markersize=3, label="GNSS Measurements")
    plt.plot(time / 60, data_GT, '--', color='green', linewidth=3, label="Ground truth")
    plt.plot(time / 60, x_hat_AM, '--', color='blue', linewidth=3, label="NUV-AM")
    plt.ylabel('Position [m]', size='35')
    plt.xlabel("Time [min]", size='35')
    plt.grid()
    plt.xticks(size=24)
    plt.yticks(size=24)
    plt.legend(fontsize='23')
    plt.savefig(vehicular+'/results/position_filters_'+direc+'_hor.pdf', bbox_inches='tight')
    plt.show()

