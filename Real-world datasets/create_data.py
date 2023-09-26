from KF_simulations import *

# """Create data - White noise acceleration (second-order model)"""
def Create_data(data, time_t, r_squared_vec, q_squared_vec):

    len_time = len(time_t)
    x_hat = np.zeros((5, 2, len_time, len(r_squared_vec), len(q_squared_vec)))
    x_init = np.array([[data[0]], [0]])
    P_init = np.diag((1, 1))

    for j in range(len(r_squared_vec)):
        print(j)
        r_squared = np.array((1)) * r_squared_vec[j]

        for i in range(len(q_squared_vec)):
            q_squared = np.array((1, 1)) * q_squared_vec[i]
            q_squared = np.expand_dims(q_squared, axis=1)

            x_hat[1, :, :, j, i] = ORKF(data, time_t, len_time, r_squared, q_squared, x_init, P_init)
            x_hat[2, :, :, j, i] = Chi_squared(data, time_t, len_time, r_squared, q_squared, x_init, P_init)
            x_hat[3, :, :, j, i] = OIKF_AM(data, time_t, len_time, r_squared, q_squared, x_init, P_init)
            x_hat[4, :, :, j, i] = OIKF_EM(data, time_t, len_time, r_squared, q_squared, x_init, P_init)
            x_hat[0, :, :, j, i] = KF(data, time_t, len_time, r_squared, q_squared, x_init, P_init)

    return x_hat
