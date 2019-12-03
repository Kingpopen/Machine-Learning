import numpy as np
import pandas as pd
#从sklearn中导入鸢尾花数据集
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


#产生数据
def loaddata():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns = iris.feature_names)
    df['label'] = iris.target
    df.columns = [
        'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
    ]

    data = np.array(df.iloc[:100, [0, 1, -1]])
    X, y = data[:, :-1], data[:, -1]
    Y = np.array([1 if i ==1 else -1 for i in y])

    print("the shape of x is:",X.shape)
    print("the shape of y is:",Y.shape)


    return X, Y


#感知机模型类
class Model:
    # 数据的初始化
    def __init__(self):
        self.lr = 0.1

    def sign(self, x, w, b):
        # the shape of temp is (N, 1)
        temp = np.dot(x, w) + b
        return temp

    # 随机梯度下降算法
    def SGD(self, X_train, Y_train):
        # the shape of x_train is (N, n) the shape of Y_train is (N, 1)
        # the shape of w is (1, n)
        self.w = np.zeros(len(X_train[0]), dtype=np.float32)
        self.b = 0
        self.x_data = X_train
        self.y_data = Y_train


        flag = True
        wrong_count = 0

        while flag:
            for i in range(len(X_train)):
                x = X_train[i]
                y = Y_train[i]
                if self.sign(x, self.w, self.b)*y <= 0:
                    wrong_count += 1
                    #更新w和b的参数
                    self.w = self.w + self.lr*x*y
                    self.b = self.b + self.lr*y

            if wrong_count == 0:
                flag = False
            wrong_count = 0
        return {'w': self.w, 'b': self.b}


    # 对偶形式的SGD
    def SGD2(self, X_train, Y_train):
        # 数据初始化
        self.x_data = X_train
        self.y_data = Y_train
        self.w = np.zeros(len(X_train[0]), dtype= np.float32)
        self.b = 0
        alpha = np.zeros(len(X_train))
        # Gram矩阵的shape is N*N
        Gram = np.dot(X_train, np.transpose(X_train))
        flag = True
        wrong_count = 0


        while flag:
            for i in range(len(X_train)):
                temp = 0
    #            判断是否为误分类点
                for j in range(len(X_train)):
                   temp += alpha[j]*Y_train[j]*Gram[i][j]

                if (temp+self.b)*Y_train[i] <= 0:
                    alpha[i] += self.lr
                    self.b += self.lr*Y_train[i]
                    wrong_count += 1

            if wrong_count == 0:
                flag = False
            wrong_count = 0

        for i in range(len(X_train)):
            self.w += alpha[i]*X_train[i]*Y_train[i]

        return {'w': self.w, 'b': self.b}





    def plot(self):
        x = np.linspace(4, 7, 100)
        y = (self.w[0]*x + self.b)/-self.w[1]
        plt.plot(x, y, color='b')
        plt.scatter(self.x_data[:50, 0], self.x_data[:50, 1], label='-1')
        plt.scatter(self.x_data[50:, 0], self.x_data[50:, 1], label='1')
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend()
        plt.show()

if __name__ == '__main__':
    X_train, Y_train = loaddata()
    model = Model()
    result = model.SGD(X_train, Y_train)
    w = result['w']
    b = result['b']
    print("w:", w)
    print("b:", b)
    # model.plot()

    result = model.SGD2(X_train, Y_train)
    print("w:", result['w'])
    print("b:", result['b'])
    model.plot()
