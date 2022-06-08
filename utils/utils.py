import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import preprocessing

from models import get_model_from_name
def standardization(data):
    mean = np.mean(data, axis=0)
    var = np.var(data, axis=0)
    data = (data-mean)/var
    return data, mean, var

def data_split(data, train_size):
    '''
    不随机的划分训练集和测试集
    :param data:待划分的数据集
    :param scale: 训练集的比例,[0,1]
    :return:
    '''
    data_train = data[:int(data.shape[0]*train_size), :]
    data_test = data[int(data.shape[0]*train_size):, :]
    return data_train, data_test


def load_data(file, eliminate=None):
    '''
    读取数据并进行标准化处理
    :param file: 数据文件的路径
    :param eliminate: 剔除某一列。从0开始，eliminate=0相当于剔除第一列特征值
    :return:
    '''
    std = preprocessing.StandardScaler()
    # ori_data = std.inverse_transform(std_data)
    data = pd.read_csv(file)
    data_array = np.asarray(data)
    columns = data.columns.values
    if eliminate != 0:
        data.drop([columns[eliminate-1]], axis=1, inplace=True)
        # print('删除的特征为：', columns[eliminate-1])
    std_data, mean, var = standardization(data_array)  # 对数据进行标准化
    # std_data = std.fit_transform(data)  # 对数据进行标准化
    train, test = data_split(std_data, train_size=0.5)
    # train, test = train_test_split(std_data, test_size=0.25, random_state=2020)
    x_train = train[:, :-1]
    y_train = train[:, -1].ravel()
    x_test = test[:, :-1]
    y_test = test[:, -1].ravel()

    # x_train, y_train = input.iloc[:, :], output.iloc[:, :]
    # x_test, y_test = input.iloc[:, :], output.iloc[:, :]
    std_cof = [mean, var]
    return x_train, x_test, y_train, y_test, std_cof, columns[eliminate]

def try_different_method(data, method, plot=True):
    # 读取数据，并划分训练集和测试集
    x_train, x_test, y_train, y_test, std_cof, feature_name = data
    # 使用训练集训练模型
    model = get_model_from_name[method]
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    # 分别使用训练样本和测试样本进行预测
    result_train = model.predict(x_train)
    result_test = model.predict(x_test)

    # 计算训练集和测试集的RMSE、MAE、R2等评价系数
    # metrics.mean_squared_error(y_test, result)
    MAE_train = metrics.mean_absolute_error(y_train, result_train)
    RMSE_train = np.sqrt(metrics.mean_squared_error(y_train, result_train))
    R2_train = metrics.r2_score(y_train, result_train)

    MAE_test = metrics.mean_absolute_error(y_test, result_test)
    RMSE_test = np.sqrt(metrics.mean_squared_error(y_test, result_test))
    R2_test = metrics.r2_score(y_test, result_test)
    # print("这三种评价指标中，RMSE和MAE越小，R2越接近1，说明训练出的模型对数据的拟合程度越好。")
    # print('训练集:\nRMSE={}, MAE={}, R2={}\n'.format(RMSE_train, MAE_train, R2_train),
    #       '测试集:\nRMSE={}, MAE={}, R2={}'.format(RMSE_test, MAE_test, R2_test))

    # 反归一化数据
    y_train = y_train * std_cof[1][-1] + std_cof[0][-1]
    y_test = y_test * std_cof[1][-1] + std_cof[0][-1]
    result_train = result_train * std_cof[1][-1] + std_cof[0][-1]
    result_test = result_test * std_cof[1][-1] + std_cof[0][-1]
    # print("预测的训练集数值为（和表中数据顺序一致）：{}".format(result_train))
    # print("预测的测试集数值为（和表中数据顺序一致）：{}".format(result_test))

    ex_name = '{}_{}'.format(method,feature_name)
    if plot:
        # print("测试集预测值：\n", result_test)
        # print("测试集真实值：\n", y_test)
        # 画图
        fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 4))
        # 画训练集的图
        plt.suptitle("method:{}\nfeature drop:{}".format(method,feature_name))
        ax0.plot(np.arange(len(result_train)), y_train, 'go-', label='true value')
        ax0.plot(np.arange(len(result_train)), result_train, 'ro-', label='predict value')
        ax0.set_title('train:\nRMSE={}\nMAE={}\nR2={}'.format(RMSE_train, MAE_train, R2_train))
        ax0.legend()
        # 画测试集的图
        ax1.plot(np.arange(len(result_test)), y_test, 'go-', label='true value')
        ax1.plot(np.arange(len(result_test)), result_test, 'ro-', label='predict value')
        ax1.set_title('test:\nRMSE={}\nMAE={}\nR2={}'.format(RMSE_test, MAE_test, R2_test))
        ax1.legend()
        fig.tight_layout()
        plt.savefig('./results/result_{}_{}.jpg'.format(method,feature_name))
        plt.close()
        # plt.show()
    return R2_test, ex_name

