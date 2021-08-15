import scipy.io as sio
import numpy as np
from flow_func import data_time_steps
import matplotlib.pyplot as plt


normal_bool = False
train_index = 1
fault_num = 1
if normal_bool:
    rute = './data'
    train_data = sio.loadmat(rute + r'\Train' + str(train_index) + '.mat')['T%d' % train_index]

    train_data = train_data[:, :-1]
    # train = train_data[np.where(np.less_equal(train_data[:, 12], 80))]
    train = train_data
    Y = train[..., 12]
    train_data = data_time_steps(np.concatenate((train[..., :12], train[..., 13:]), axis=-1))
    Y = data_time_steps(Y, label_data=True)

    sio.savemat('./data' + '/T%d_dynamic.mat' % train_index, {'X': train_data})
    sio.savemat('./data' + '/T%d_dynamic_y.mat' % train_index,
                {'Y': Y})

else:
    rute = './data'
    train_data = sio.loadmat(rute + '/CAO_threephaseFaultyCase%d' % fault_num + '.mat')['Set%d_2' % fault_num]
    train = train_data[:, :-1]
    Y = train[..., 12]
    train_data = data_time_steps(np.concatenate((train[..., :12], train[..., 13:]), axis=-1))
    Y = data_time_steps(Y, label_data=True)
    # train_data = data_time_steps(train)

    sio.savemat('./data' + '/fault%d_dynamic.mat' % fault_num, {'X': train_data})
    sio.savemat('./data' + '/fault%d_dynamic_y.mat' % fault_num,
                {'Y': Y})

    # for i in range(23):
    plt.plot(Y)
    plt.show()

