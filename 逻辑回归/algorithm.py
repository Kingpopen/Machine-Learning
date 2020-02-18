import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


LR = 0.01
Step = 10000


# 获取鸢尾花数据
def create_Data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, :])
    return data[:, :-1], data[:, -1]


# sigmoid 函数
def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 损失函数
def Loss_function(x, y, W, b):
    total_loss = 0
    for x_i, y_i in zip(x, y):
        temp = np.dot(np.transpose(W), x_i) + b
        y_hat = Sigmoid(temp)
        loss = -1 * (y_i * np.log2(y_hat) + (1 - y_i) * np.log2(1-y_hat))
        total_loss += loss
    return total_loss/len(x)


#优化器(使用梯度下降法的地方，默认使用的是SGD)
def Optimizer(x, y, W, b, LR = 0.001, Type = 'SGD'):
    #SGD
    if Type == 'SGD':
        index = np.array([i for i in range(len(x))])  #将数据打乱
        for i in index:
            x_i, y_i = x[i], y[i]
            y_hat = Sigmoid(np.dot(np.transpose(W), x_i) + b)
            W = W - LR * (y_hat - y_i) * x_i
            b = b - LR * (y_hat - y_i)
    return W, b


# 训练
def Train(X, Y):
    #初始化W和b
    W = np.random.randn(X.shape[1])
    b = 0
    for step in range(Step):
        W, b = Optimizer(X, Y, W, b, LR)
        if step % 1000 == 0:
            loss = Loss_function(X, Y, W, b)
            print("Loss is :", loss)

    parameter = {'W': W,
                 'b': b}
    return parameter

#预测
def Predict(X, Y, parameter):
    W = parameter['W']
    b = parameter['b']
    total = len(X)
    cnt = 0
    for x, y in zip(X, Y):
        #预测值>=0.5 归为label =1，预测值<0.5，归为label = 0
        y_hat = 1 if Sigmoid(np.dot(np.transpose(W), x) + b) >= 0.5 else 0
        if y_hat == y:
            cnt += 1
    print("ACC:", cnt/total)



if __name__ == '__main__':
    X, Y = create_Data()
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.3)
    print("the number of train:", len(train_X))
    print("the number of test:", len(test_X))

    parameter = Train(train_X, train_Y)
    Predict(test_X, test_Y, parameter)



