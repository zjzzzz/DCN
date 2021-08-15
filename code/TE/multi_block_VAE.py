import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from VAE_func import *


def main_func(train_bool, save_name, fault_num, show_regress, rute):
    df_train, df_train_y = read_data(0, True)
    df_ver, df_ver_y = read_data(0, False)
    df_test, _ = read_data(fault_num, True)

    trans = StandardScaler()
    np_nor = trans.fit_transform(df_train)
    np_ver = trans.transform(df_ver)
    np_test = trans.transform(df_test)

    trans_y = StandardScaler()
    np_nor_y = trans_y.fit_transform(df_train_y)
    np_ver_y = trans_y.transform(df_ver_y)

    vars = []
    vars.append([i - 1 for i in [1, 2, 3, 7, 8, 24, 25, 11, 13, 10, 28]])
    vars.append([i - 1 for i in [10, 11, 13, 20, 27, 28]])
    vars.append([i - 1 for i in [11, 13, 10, 28, 16, 18, 19, 31]])

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
    np_nor_y = data_time_steps(np_nor_y, label_data=True)
    np_ver_y = data_time_steps(np_ver_y, label_data=True)

    fig1 = plt.figure(100, dpi=300)
    mse_train_ver = []
    for i in range(len(vars)):
        print('{} / {} block'.format(i+1, len(vars)))
        train_data = np_nor[:, vars[i]]
        ver_data = np_ver[:, vars[i]]
        test_data = np_test[:, vars[i]]

        train_data = data_time_steps(train_data)
        ver_data = data_time_steps(ver_data)
        test_data = data_time_steps(test_data)

        encoder, vae = construct_cnn_vae_model(i, train_data.shape[-1])

        if train_bool:
            history = train_model(rute, vae, i, train_data, np_nor_y, ver_data, np_ver_y, save_name)
        else:
            train_kl_loss_y, fault_kl_loss_y, model = test_model(rute, encoder, vae, i, train_data, test_data, save_name)

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
                # plt.ylim([0,50])
                plt.legend(['ver_y', 'predict_y'])
                plt.show()

            control_line = compute_threshold(train_kl_loss_y, bw=None, alpha=0.99)

            contri_x_b, contri_y_b = contribute_plot(rute, encoder, vae, i, train_data, test_data, vars, save_name)
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
        pt2_n = pt2_n / sum(pt2_x_n)
        pt2_f = pt2_f / sum(pt2_x_f)

        n_bay = pt2_f.shape[0]
        plt.figure(2, dpi=300)
        # plt.subplot(211)
        plt.plot(pt2_f)
        plt.plot(np.ones(n_bay) * sig_level, 'r--')
        # plt.ylim([0, 0.1])
        plt.title('{} KL statistic of test data in bayes'.format(save_name.upper()))
        plt.show()

        plt.figure(3, dpi=300)
        plt.bar(range(1, len(contri_x)+1), contri_y)
        plt.xticks(range(1, len(contri_x)+1), contri_x)
        plt.text(5, max(contri_y)-5, 'block 1')
        plt.text(14, max(contri_y)-5, 'block 2')
        plt.text(21, max(contri_y)-5, 'block 3')
        plt.vlines(11.5, 0, max(contri_y), colors="r", linestyles="dashed")
        plt.vlines(17.5, 0, max(contri_y), colors="r", linestyles="dashed")
        plt.title('multi-block DCN based contribution plot')
        plt.show()


if __name__ == '__main__':
    show_regress = False
    rute = './models'
    model_name = 'cnn_vae'
    fault_list = [1,2,3]
    rate = []
    for fault_num in fault_list:
        main_func(train_bool=False, save_name=model_name, fault_num=fault_num,
                                                     show_regress=show_regress, rute=rute)

