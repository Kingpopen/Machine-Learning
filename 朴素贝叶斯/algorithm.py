import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter

# 获取鸢尾花数据
def create_Data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, :])
    return data[:, :-1], data[:, -1]

# 获取朴素贝叶斯数据
def create_Data_naive():
    X_train = np.array([[1, 'S'], [1, 'M'], [1, 'M'], [1, 'S'], [1, 'S'],
               [2, 'S'], [2, 'M'], [2, 'M'], [2, 'L'], [2, 'L'],
               [3, 'L'], [3, 'M'], [3, 'M'], [3, 'L'], [3, 'L']])
    X_test = np.array([[2, 'S']])
    y_train = np.array([-1, -1, 1, 1, -1,
                        -1, -1, 1, 1, 1,
                        1, 1, 1, 1, -1])
    y_test = np.array([-1])
    return X_train, X_test, y_train, y_test

# 高斯朴素贝叶斯类
class GaussianNB:
    def __init__(self):
        self.model = None
        self.train_X = None
        self.train_Y = None
        self.num_class = 0
        self.keys_y = None
        self.prior = None
        self.avg = 0 #训练集各特征均值 [x(1)的均值,x(2)的均值,x(3)的均值...] x(1), x(2)...分别代表不同的特征
        self.dev = 0 #训练集各特征方差 [x(1)的方差,x(2)的方差,x(3)的方差...] x(1), x(2)...分别代表不同的特征

    # 计算期望(用均值代替),条件期望，条件用label表示
    def mean(self, label):
        return np.array([self.train_X[label == i].mean(axis=0) for i in self.keys_y])

#     计算方差(用均值近似代替)
    def stdev(self, label):
        return np.sqrt(np.array([self.train_X[label == i].var(axis=0) for i in self.keys_y]))

    # 计算概率密度函数公式
    def Get_probability(self, x):
        result = []
        for i in range(len(self.avg)):
            exp = np.exp(-np.power((x - self.avg[i]), 2) / (2 * np.power(self.dev[i], 2)))
            pdf = 1 / (np.sqrt(2 * np.pi) * self.dev[i]) * exp
            result.append(pdf)
        return np.array(result)

    # 计算先验概率
    def Get_prior(self):
        cnt_Y = Counter(self.train_Y)
        total_num = len(self.train_Y)
        self.prior = np.array([cnt_Y[i] / total_num for i in self.keys_y])

    #训练模型
    def train(self, X, Y):
        cnt_y = Counter(Y)
        self.train_X = X
        self.train_Y = Y
        self.num_class = len(cnt_y)
        self.keys_y = list(cnt_y.keys())
        self.avg = self.mean(Y)
        self.dev = self.stdev(Y)
        self.Get_prior()

    # 评分函数
    def Score(self, X, Y):
        num_test = len(X)
        acc = 0
        for i in range(num_test):
            data = X[i]
            prob_condition = self.Get_probability(data)
            likeli = np.product(prob_condition, axis=1)
            prob = list(likeli * self.prior)
            index = prob.index(max(prob))
            if self.keys_y[index] == Y[i]:
               acc += 1

        acc = acc / num_test
        print("acc:", acc)

    # 预测函数
    def Predict(self, X):
        num_test = len(X)
        result = []
        for i in range(num_test):
            data = X[i]
            prob_condition = self.Get_probability(data)
            likeli = np.product(prob_condition, axis=1)
            prob = list(likeli * self.prior)
            index = prob.index(max(prob))
            result.append(self.keys_y[index])
        return result

# 朴素贝叶斯
class NaiveBayes1:
    def __init__(self):
        # 先验概率的值
        self.prior = None
        self.likeli = None
        self.keys_y = None
        self.keys_x = None
    # 获取概率(先验概率，后验概率等)
    '''
    默认Y只有一个维度的特征
    P(Y|X) = P(X|Y)*P(Y)/P(X)
    P(X|Y)为似然
    P(Y)为先验概率
    '''
    def train(self, X_train, y_train):
        # 求先验概率
        count = 0
        dic_y = {}
        cnt_Y = Counter(y_train)
        keys_y = list(cnt_Y.keys())
        self.prior = np.array([cnt_Y[keys_y[i]] / len(y_train) for i in range(len(keys_y))])
        # 建立Y的索引（dic_y{'0':y1的名字，'1':y2的名字,...}）
        for key in keys_y:
            dic_y[count] = key
            count += 1
        # print("dic_y:", dic_y)
        self.keys_y = keys_y


        # 求似然
        feature_X = len(X_train[0]) #得到x的特征种类的个数
        # print("feature_X:", feature_X)
        list_index = [] #[{x1特征的索引},{x2特征的索引}...]
        list_likli = [] #[{y1为条件的似然},{y2为条件的似然}...]


        # 建立x的索引并且求似然
        for k in range(len(keys_y)):
            list_y = []   #[yi为条件的似然]
            count_y = cnt_Y[self.keys_y[k]] #yk的个数
            list_keys_x = []

            # print("在Y轴上以%s为条件" % keys_y[k])
            for i in range(feature_X):
                list_x = []
                dic_x = {}
                cnt_x = Counter(X_train[:, i])
                keys_x = list(cnt_x.keys())
                list_keys_x.append(keys_x)
                count = 0
                # 在此处建立索引计算似然
                for j in keys_x:
                    dic_x[j] = count
                    count += 1
                    # 似然
                    num = 0 #统计分子的个数
                    for z in zip(X_train[:, i], y_train[:]):
                        if z[0] == j and z[1] == keys_y[k]:
                            num += 1
                    pro = num / count_y
                    list_x.append(pro)
                    # print("x%d=%s的概率为：%d/%d" % (i, j, num , count_y))
                # print("list_x:", list_x)
                list_y.append(list_x)
                list_index.append(dic_x)
            list_likli.append(list_y)

        self.likeli = list_likli
        self.keys_x = list_keys_x

        # print("list_likli:", list_likli)
        # print("list_keys_x:", list_keys_x)
    # 预测
    def Predict(self, x):
        feature_x_num = len(self.keys_x)
        indexs = []
        prob = []
        for n in range(len(x)):
            data = x[n]
            index = []
            for i in range(feature_x_num):
                keys = self.keys_x[i]
                num = len(keys)
                for j in range(num):
                    if keys[j] == data[i]:
                        index.append(j)

            indexs.append(index)

        # print("x:", x)
        # print("index:", indexs)
        # print("prob:", self.likeli)
        feature_y_num = len(self.likeli)
        # print("len of likeli:", feature_y_num)

        for i in range(feature_y_num):
            score = 1
            for j in range(feature_x_num):
                score *= self.likeli[i][j][index[j]]
            score = score * self.prior[i]
            prob.append(score)
        index = prob.index(max(prob))

        print("label:", self.keys_y[index])

# 朴素贝叶斯的实例
def example_bayes():
    X_train, X_test, y_train, y_test = create_Data_naive()
    model = NaiveBayes1()
    model.train(X_train, y_train)
    model.Predict(X_test)


if __name__ == '__main__':
    X, y = create_Data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = GaussianNB()
    model.train(X_train, y_train)
    model.Score(X_test, y_test)











