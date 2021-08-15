#%%
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from flow_func import *
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print('memory grouth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
else:
    print('No enough GPU')


def main_func(train_bool=False, show_regress=False, fault_num=1, save_name='cnn_vae', filters=10, latent=60):
    rute = './model'

    df_train, df_train_y = read_flow_data(1, True)
    time_step = df_train.shape[1]
    variable_num = df_train.shape[2]

    df_test, df_test_y = read_flow_data(fault_num, False)

    df_train = np.reshape(df_train, (-1, df_train.shape[-1]))
    df_test = np.reshape(df_test, (-1, df_test.shape[-1]))

    trans = StandardScaler()
    np_nor = trans.fit_transform(df_train)
    np_test = trans.transform(df_test)

    trans_y = StandardScaler()
    np_nor_y = trans_y.fit_transform(df_train_y)
    np_nor_y = np_nor_y

    np_nor = np.reshape(np_nor, (-1, time_step, variable_num))
    train_size = int(np_nor.shape[0] * 0.8)
    np_ver = np_nor[train_size:, ...]
    np_ver_y = np_nor_y[train_size:]
    np_test = np.reshape(np_test, (-1, time_step, variable_num))

    vars = []
    vars.append([i - 1 if i < 13 else i - 2 for i in [1,8,21,22,9,15,18,23,3,6,10,16,2]])
    vars.append([i - 1 if i < 13 else i - 2 for i in [3,6,10,16,2,4,7,11,12,14,17]])
    vars.append([i - 1 if i < 13 else i - 2 for i in [4,7,11,12,14,17,5,20,19]])
    sort_vars = []
    for var in vars:
        var.sort()
        sort_vars.append(var)
    vars = sort_vars

    beta = 0.99
    sig_level = 0.01

    pt2_x_f = []
    pt2_f = 0

    pt2_x_n = []
    pt2_n = 0

    contri_x = []
    contri_y = []
    fig1 = plt.figure(100, dpi=300)
    mse_train_ver = []
    for i in range(len(vars)):
        print('{} / {} block'.format(i+1, len(vars)))
        train_data = np_nor[..., vars[i]]
        ver_data = np_ver[..., vars[i]]
        test_data = np_test[..., vars[i]]
        encoder, vae = construct_cnn_vae_model(i, train_data.shape[-1], filters=filters, latent=latent)

        if train_bool:
            history = train_model(rute, vae, i, train_data, np_nor_y, ver_data, np_ver_y, save_name, latent=latent, filter_num=filters)
        else:
            train_kl_loss_y, fault_kl_loss_y, model = test_model(rute, encoder, vae, i, train_data, test_data, save_name, latent=latent, filter_num=filters)
            predict_train_y = vae.predict(train_data)
            predict_ver_y = vae.predict(ver_data)

            mse_train = sum((predict_train_y - np_nor_y) ** 2) / predict_train_y.shape[0]
            mse_ver = sum((predict_ver_y - np_ver_y)**2) / predict_ver_y.shape[0]
            mse_train_ver.append([mse_train, mse_ver])

            if show_regress:
                plt.figure(7)
                plt.subplot(211)
                plt.plot(range(np_nor_y.shape[0]), np_nor_y)
                plt.plot(range(predict_train_y.shape[0]), predict_train_y)
                plt.title('train set block {} '.format(i+1))
                plt.legend(['train_y', 'predict_y'])
                plt.subplot(212)
                plt.plot(range(np_ver_y.shape[0]), np_ver_y)
                plt.plot(range(predict_ver_y.shape[0]), predict_ver_y)
                plt.title('ver set block {} '.format(i+1))
                plt.legend(['ver_y', 'predict_y'])
                plt.show()

            control_line = compute_threshold(train_kl_loss_y, bw=None, alpha=0.99)

            contri_x_b, contri_y_b = contribute_plot(rute, encoder, vae, i, train_data, test_data, vars, save_name, latent=latent, filter_num=filters)
            contri_x_b = [v+1 if v > 12 else v for v in list(map(int, contri_x_b))]
            contri_x.extend(contri_x_b)
            contri_y.extend(contri_y_b)

            pt2_x_n_b, pt2_n_x_b = cal_bayes(train_kl_loss_y, control_line, sig_level)
            pt2_x_n.append(pt2_x_n_b)
            pt2_n += np.multiply(pt2_x_n_b, pt2_n_x_b)

            pt2_x_f_b, pt2_f_x_b = cal_bayes(fault_kl_loss_y, control_line, sig_level)
            pt2_x_f.append(pt2_x_f_b)
            pt2_f += np.multiply(pt2_x_f_b, pt2_f_x_b)

            n = fault_kl_loss_y.shape[0]
            ax1 = fig1.add_subplot(len(vars), 1, i + 1)
            ax1.plot(fault_kl_loss_y)
            ax1.plot(np.ones(n) * control_line, 'r--')
            ax1.set_title('{} KL statistic of fault {} in block {}'.format(save_name.upper(), fault_num, i + 1))

    if not train_bool:
        if len(vars) > 1:
            pt2_n = pt2_n / sum(pt2_x_n)
            pt2_f = pt2_f / sum(pt2_x_f)

            n_bay = pt2_f.shape[0]
            plt.figure(2, dpi=300)
            plt.plot(pt2_f)
            plt.plot(np.ones(n_bay) * sig_level, 'r--')
            plt.ylim([0, 0.1])
            plt.title('{} KL statistic of test data in bayes'.format(save_name.upper()))
            plt.show()

        plt.figure(3, dpi=300, figsize=(8,5))
        plt.bar(range(1, len(contri_x) + 1), contri_y)
        plt.xticks(range(1, len(contri_x) + 1), contri_x)
        plt.text(6, max(contri_y) - 5, 'block 1')
        plt.text(18, max(contri_y) - 5, 'block 2')
        plt.text(26, max(contri_y) - 5, 'block 3')
        plt.vlines(13.5, 0, max(contri_y), colors="r", linestyles="dashed")
        plt.vlines(24.5, 0, max(contri_y), colors="r", linestyles="dashed")
        plt.title('multi-block DCN based contribution plot')
        plt.show()


if __name__ == '__main__':
    main_func(train_bool=False, save_name='cnn_vae', fault_num=1, show_regress=False)

