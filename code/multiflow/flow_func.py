from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense, Conv1D, Flatten
from keras.models import Model
from keras.losses import mse
from keras import backend as K
from keras.optimizers import Adam
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--activa', type=str, default='selu')
parser.add_argument('--variables', type=int, default=22)
parser.add_argument('--time_steps', type=int, default=15)
parser.add_argument('--filters', type=int, default=10)
parser.add_argument('--latent', type=int, default=60)
parser.add_argument('--mse_beta', type=int, default=10)
parser.add_argument('--beta', type=int, default=1)
parser.add_argument('--l1_x', type=float, default=0.0)
args = parser.parse_args()
print(args)


def sampling(arg):
    z_mean, z_log_var = arg
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def read_flow_data(error=0, is_train=True):
    if is_train:
        path_x = './data/T%d_dynamic.mat' % error
        path_y = './data/T%d_dynamic_y.mat' % error
    else:
        path_x = './data/fault%d_dynamic.mat' % error
        path_y = './data/fault%d_dynamic_y.mat' % error

    data_x = sio.loadmat(path_x)['X']
    data_y = sio.loadmat(path_y)['Y'].T
    return data_x, data_y


def data_time_steps(data, time_steps=args.time_steps, label_data=False):
    if label_data:
        return data[time_steps-1:]
    else:
        nrow = data.shape[0]
        data_time = []
        for i in range(time_steps):
            data_time.append(data[i:nrow-time_steps+1+i])
        return np.array(data_time).transpose((1, 0, 2))


def compute_threshold(data, bw=None, alpha=0.99):
    data = data.reshape(-1,1)
    Min = np.min(data)
    Max = np.max(data)
    Range = Max-Min
    x_start = Min-Range
    x_end = Max+Range
    nums = 2**15
    dx = (x_end-x_start)/(nums-1)
    data_plot = np.linspace(x_start, x_end, nums)
    if bw is None:
        data_median = np.median(data)
        new_median = np.median(np.abs(data-data_median))/0.6745 + 0.00000001
        bw = new_median*((4/(3*data.shape[0]))**0.2)
    kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(data.reshape(-1,1))
    log_pdf = kde.score_samples(data_plot.reshape(-1, 1))
    pdf = np.exp(log_pdf)
    CDF = 0
    index = 0

    while CDF <= alpha:
        CDF += pdf[index]*dx
        index += 1

    return data_plot[index]


def kl_loss_cal(test_mean, test_log_var):
    loss = -0.5 * np.sum(1 + test_log_var - np.square(test_mean) - np.exp(test_log_var), axis=-1)
    return loss


def kl_loss_cal_test(test_mean, test_log_var):
    loss = -0.5 * (1 + test_log_var - np.square(test_mean) - np.exp(test_log_var))
    return loss


def construct_cnn_vae_model(block, input_shape, time_steps=args.time_steps, filters=args.filters, latent=args.latent):
    def vae_loss(x, x_decoded_mean):
        xent_loss = mse(x, x_decoded_mean)
        kl_loss_y = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(args.mse_beta * xent_loss + args.beta * kl_loss_y)

    def kl_loss(x, x_decoded_mean):
        kl_loss_y = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(kl_loss_y)

    def mse_loss(x, x_decoded_mean):
        mse_loss_y = mse(x, x_decoded_mean)
        return K.mean(mse_loss_y)


    inputs = Input(shape=(time_steps, input_shape,), name='encoder_input_{}'.format(block))
    conv1 = Conv1D(filters=filters, kernel_size=3, strides=1, activation=args.activa, name='pre_conv1_{}'.format(block))(inputs)
    conv2 = Conv1D(filters=filters, kernel_size=3, strides=1, activation=args.activa, name='pre_conv2_{}'.format(block))(conv1)
    pre_z = Flatten(name='pre_z_{}'.format(block))(conv2)

    z_mean = Dense(latent, activation=args.activa, name='z_mean_{}'.format(block))(pre_z)
    z_log_var = Dense(latent, activation=args.activa, name='z_log_var_{}'.format(block))(pre_z)

    z = Lambda(sampling, output_shape=(latent,), name='latent_{}'.format(block))([z_mean, z_log_var])

    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder_{}'.format(block))
    encoder.summary()

    output_y = Dense(1, activation='linear', name='output_y_{}'.format(block))(z)

    vae = Model(inputs, output_y, name='vae_{}'.format(block))

    optimizer = Adam(lr=args.lr)
    vae.compile(optimizer=optimizer, loss={'output_y_{}'.format(block): vae_loss},
                metrics=[kl_loss, mse_loss])

    vae.summary()
    return encoder, vae


def train_model(rute, vae, block, train_x, train_y, ver_x, ver_y, save_name, latent=args.latent, l1_x=args.l1_x, filter_num=args.filters, epochs=args.epochs, batch_size=args.batch_size):
    print('--------------------train VAE_{}--------------------'.format(block))
    history_vae = vae.fit(train_x, train_y, validation_data=(ver_x, ver_y),
                          epochs=epochs,
                          batch_size=batch_size, shuffle=True)
    plt.figure(2)
    plt.plot(history_vae.history['loss'])
    plt.plot(history_vae.history['val_loss'])
    plt.title('Model {} loss'.format(block))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    print('train over')
    vae.save_weights(rute + '/VAE_TE_b{}_latent{}_l1{}_filter{}_{}.h5'.format(block, latent, l1_x, filter_num, save_name))

    predict_train_y = vae.predict(train_x)
    predict_ver_y = vae.predict(ver_x)

    plt.figure(7)
    plt.subplot(211)
    plt.plot(range(train_y.shape[0]), train_y)
    plt.plot(range(predict_train_y.shape[0]), predict_train_y)
    plt.title('train set block {} '.format(block))
    plt.legend(['train_y', 'predict_y'])
    plt.subplot(212)
    plt.plot(range(ver_y.shape[0]), ver_y)
    plt.plot(range(predict_ver_y.shape[0]), predict_ver_y)
    plt.title('ver set block {} '.format(block))
    plt.legend(['ver_y', 'predict_y'])
    plt.show()

    return history_vae


def test_model(rute, encoder, vae, block, train_x, fault_x, save_name, latent=args.latent, l1_x=args.l1_x, filter_num=args.filters):
    vae.load_weights(rute + r'/VAE_TE_b{}_latent{}_l1{}_filter{}_{}.h5'.format(block, latent, l1_x, filter_num, save_name))

    weights = vae.get_weights()
    l1_weight = weights[-2]

    train_encoder_mean, train_encoder_log_var, train_encoder = encoder.predict(train_x)

    train_mean_y = train_encoder_mean
    train_var_y = train_encoder_log_var

    train_kl_loss_y = kl_loss_cal(train_mean_y, train_var_y)

    fault_encoder_mean, fault_encoder_log_var, fault_encoder = encoder.predict(fault_x)

    fault_mean_y = fault_encoder_mean
    fault_var_y = fault_encoder_log_var

    fault_kl_loss_y = kl_loss_cal(fault_mean_y, fault_var_y)

    return train_kl_loss_y, fault_kl_loss_y, vae


def contribute_plot(rute, encoder, vae, i, train_data, test_data, vars, save_name, latent=args.latent, filter_num=args.filters):
    axis_x = []
    axis_y = []
    for t in range(len(vars[i])):
        print('The contribute of variable {0} in block {1}'.format(vars[i][t]+1, i))
        train_data_var = np.zeros_like(train_data)
        test_data_var = np.zeros_like(test_data)

        train_data_var[:, :, t] = train_data[:, :, t]
        test_data_var[:, :, t] = test_data[:, :, t]

        train_kl_loss_y, fault_kl_loss_y, _ = test_model(rute, encoder, vae, i, train_data_var, test_data_var, save_name, latent=latent,filter_num=filter_num)
        train_kl_loss_y = train_kl_loss_y.reshape(-1, 1)
        fault_kl_loss_y = fault_kl_loss_y.reshape(-1, 1)

        trans = StandardScaler()
        trans.fit(train_kl_loss_y)
        fault_kl_loss_y = trans.transform(fault_kl_loss_y)

        fault_kl_loss_y_mean = abs(np.mean(fault_kl_loss_y))
        axis_x.append(str(vars[i][t] + 1))
        axis_y.append(fault_kl_loss_y_mean)

    return axis_x, axis_y


def cal_bayes(t2, t2_line, sig_level):
    Pt2_X_N = np.exp(-t2 / t2_line)
    Pt2_X_F = np.exp(-t2_line / (t2 + 0.0000000001))

    Pt2_N = 1 - sig_level
    Pt2_F = sig_level
    Pspe_N = 1 - sig_level
    Pspe_F = sig_level

    Pt2_X = Pt2_X_N * Pt2_N + Pt2_X_F * Pt2_F

    Pt2_F_X = Pt2_X_F * Pt2_F / Pt2_X

    return Pt2_X_F, Pt2_F_X


if __name__ == '__main__':
    print(1)
