import pandas as pd

from utils.utils import *
from models import get_model_from_name

if __name__ == '__main__':
    #--------------------#
    #   data_path
    #   数据文件地址
    #--------------------#
    data_path = '../data/t-d_data.csv'
    # --------------------#
    #   plot
    #   是否画图 (画图设置为True；不画图输出R2结果设置为False)
    #   实验结果图保存在tools/results文件夹内
    #   该文件夹还会保存所有模型的R2结果，每个模型都会遍历删除单个特征
    #   例如('RandomForest_E', -1.969445079014971)表示使用随机森林博士，删除“E”列特征，得到R2结果为-1.969445079014971
    #   值得注意的是，'LinearRegression_ΔE(eV)',代表了使用线性回归模型，没有删除任何列（ΔE(eV)为最后一列列名）。
    # --------------------#
    plot = True



    feature_num = pd.read_csv(data_path).shape[1]
    R2_rank = {}
    for method in get_model_from_name:
        for i in range(feature_num):
            data = load_data(data_path, i)
            R2_value, ex_name = try_different_method(data, method, plot=True)
            R2_rank[ex_name] = R2_value
    R2_rank_sort = sorted(R2_rank.items(), key=lambda x: x[1])
    a = list(R2_rank_sort)
    print('R2最高的前10个组合是：',list(R2_rank_sort)[-10:])
    fileObject = open('./results/0_result_R2排序(从底到高).txt', 'w')
    for ip in a:
        fileObject.write(str(ip))
        fileObject.write('\n')
    fileObject.close()

