####1.决策树回归####
from sklearn import tree

model_DecisionTreeRegressor = tree.DecisionTreeRegressor()
####2.线性回归####
from sklearn import linear_model

model_LinearRegression = linear_model.LinearRegression()
####3.SVM回归####
from sklearn import svm

model_SVR = svm.SVR()
####4.KNN回归####
from sklearn import neighbors

model_KNeighborsRegressor = neighbors.KNeighborsRegressor()
####5.随机森林回归####
from sklearn import ensemble

model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)  # 这里使用20个决策树
####6.Adaboost回归####
from sklearn import ensemble

model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=50)  # 这里使用50个决策树
####7.GBRT回归####
from sklearn import ensemble

model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)  # 这里使用100个决策树
####8.Bagging回归####
from sklearn.ensemble import BaggingRegressor

model_BaggingRegressor = BaggingRegressor()
####9.ExtraTree极端随机树回归####
from sklearn.tree import ExtraTreeRegressor

model_ExtraTreeRegressor = ExtraTreeRegressor()
####10.ANN人工神经网络回归####
from sklearn.neural_network import MLPRegressor

model_ANN = MLPRegressor(solver='adam', hidden_layer_sizes=(50, 50), activation='relu',
                         max_iter=5000)  # 两个隐藏层，每层10个节点，训练50000次
####11.GBM回归####
from sklearn.ensemble import GradientBoostingRegressor

model_GBM = GradientBoostingRegressor()

get_model_from_name = {
    "DecisionTree": model_DecisionTreeRegressor,  # DecisionTree 决策树
    "LinearRegression": model_LinearRegression,  # LinearRegression 线性回归
    "SVM": model_SVR,  # SVM
    "KNN": model_KNeighborsRegressor,  # KNN K近邻算法
    "RandomForest": model_RandomForestRegressor, # RandomForestRegressor 随机森林
    "AdaBoost": model_AdaBoostRegressor, # AdaBoost
    "GBRT": model_GradientBoostingRegressor, # GradientBoosting 梯度提升算法
    "Bagging": model_BaggingRegressor, #
    "ExtraTree": model_ExtraTreeRegressor, # 计算随机树
    "ANN": model_ANN,  # 人工神经网络
    "GBM": model_GBM,  #
}
