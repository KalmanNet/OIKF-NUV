import numpy as np
import os.path
from scipy.stats import bernoulli,rayleigh

def create_GT(N, del_t, q_2, time_t):

    data_GT = np.zeros((2, len(time_t), N))
    for n in range(N):
            q_squared = np.array([[1], [1]]) * q_2
            for t in range(1, len(time_t)):
                F = np.array([[1, del_t], [0, 1]])
                Q = np.array([[(1 / 3) * (del_t ** 3), (1 / 2) * (del_t ** 2)], [(1 / 2) * (del_t ** 2), (del_t)]]) * q_squared
                e_t = np.random.multivariate_normal([0,0], Q)
                data_GT[:, t:t+1, n] = F @ data_GT[:, t-1:t, n] + np.expand_dims(e_t, axis=1)

    return data_GT


def Data_Generation(p, scl):
    N = 200
    end_time = 10
    del_t = 1e-1
    time_t = np.linspace(0, end_time, int(end_time / del_t) + 1)
    q_squared = 1e-1
    r_squared_vec = np.array((1e1, 1, 1e-1, 1e-2, 1e-3))
    len_r_vec = len(r_squared_vec)

    file_name_GT = 'results/data_GT.npy'
    #### ground truth ########
    if os.path.isfile(file_name_GT)==False:
        data_GT = create_GT(N, del_t, q_squared, time_t)
        np.save(file_name_GT, data_GT)
        "NO DATA GT"
    else:
        data_GT = np.load(file_name_GT)
        "YES DATA GT"

    #### signal of ovseravations ########
    file_name_data_y = 'results/data_y.npy'
    if os.path.isfile(file_name_data_y) == False:

        data_y = np.zeros((data_GT.shape[0], 2, data_GT.shape[1], len_r_vec, N))
        for n in range(N):

            for r in range(len_r_vec):
                    data_y[:, 0, :, r, n] = data_GT[:, :, n] + np.random.randn(2, len(time_t)) * np.sqrt(r_squared_vec[r])
                    data_y[:, 1, :, r, n] = data_y[:, 0, :, r, n]
        np.save(file_name_data_y, data_y)
    else:
        data_y = np.load(file_name_data_y)

    for n in range(N):
        for r in range(len_r_vec):

            outliers_pos = bernoulli.rvs(p, size=len(time_t))
            index_pos = np.where(outliers_pos == 1)[0]
            ray_pos = rayleigh.rvs(size=len(index_pos), scale=scl)
            data_y[0, 1, index_pos, r, n] = data_y[0, 0, index_pos, r, n] + ray_pos

            outliers_vel = bernoulli.rvs(p, size=len(time_t))
            index_vel = np.where(outliers_vel == 1)[0]
            ray_vel = rayleigh.rvs(size=len(index_vel), scale=scl)

            data_y[1, 1, index_vel, r, n] = data_y[1, 0, index_vel, r, n] + ray_vel

    file_name_data_y_outliers = 'results/data_y_p='+str(p)+'_scl='+str(scl)
    np.save(file_name_data_y_outliers, data_y)





