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

  ############################################################
    def int2str(time):
        return str(np.round(time, 2))

    table = [[ "",                 "RMSE[m]",    " MSE[dB]",                               ],
          ['Noise floor', int2str(P_MSE_m[5, int(index_vec[5, 0]), int(index_vec[5, 1])]),  int2str(P_MSE_dB[5, int(index_vec[5, 0]),int(index_vec[5, 1])])],
          ['KF',          int2str(P_MSE_m[0, int(index_vec[0, 0]), int(index_vec[0, 1])]),  int2str(P_MSE_dB[0, int(index_vec[0, 0]),int(index_vec[0, 1])])],
          ['ORKF',        int2str(P_MSE_m[1, int(index_vec[1, 0]), int(index_vec[1, 1])]),  int2str(P_MSE_dB[1, int(index_vec[1, 0]),int(index_vec[1, 1])])],
          ['Chi-squared', int2str(P_MSE_m[2, int(index_vec[2, 0]), int(index_vec[2, 1])]),  int2str(P_MSE_dB[2, int(index_vec[2, 0]),int(index_vec[2, 1])])],
          ['OIKF-AM',     int2str(P_MSE_m[3, int(index_vec[3, 0]), int(index_vec[3, 1])]),  int2str(P_MSE_dB[3, int(index_vec[3, 0]),int(index_vec[3, 1])])],
          ['OIKF-EM',     int2str(P_MSE_m[4, int(index_vec[4, 0]), int(index_vec[4, 1])]),  int2str(P_MSE_dB[4, int(index_vec[4, 0]),int(index_vec[4, 1])]),]]

    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid', numalign='center'))
    with open(vehicular+'/results/Tab_MSE_'+direc+'.txt', 'w') as f:
        f.write(tabulate(table, headers='firstrow', tablefmt='latex', numalign='center'))
    f.close()
