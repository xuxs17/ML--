# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import matplotlib as mpl
import matplotlib.pyplot as plt


if __name__ == '__main__':

    #设置数据文件路径
    os.chdir('C:/Users/xiangshang/Desktop/LR')
    # 横向最多显示多少个字符，便于页面显示
    pd.set_option('display.width', 300)
    # 显示的最大行数和列数，如果超额就显示省略号，这个指的是多少个dataFrame的列，如果比较多又不允许换行，就会显得很乱
    pd.set_option('display.max_columns', 300)
    # 读取数据，header = None表示不使用第一行作为dataframe的列索引
    data = pd.read_csv('car.data', header=None)
    # 将列数赋给n_columns
    n_columns = len(data.columns)
    # 指定一个list
    columns = ['buy', 'maintain', 'doors', 'persons', 'boot', 'safety', 'accept']
    # 将指定的list与列长度做一个字典
    new_columns = dict(list(zip(np.arange(n_columns), columns)))
    # 将该字典作为数据的列命名
    data.rename(columns=new_columns, inplace=True)
    # 取出数据前10条看看情况
    print(data.head(10))


    # one-hot编码:有些字段的值有很多个，不便于直接使用，换成数字编码比较好，例如汽车品牌等

    # 命名一个空的dataframe
    x = pd.DataFrame()

    for col in columns[:-1]:
        # 实现one—hot的方式
        t = pd.get_dummies(data[col])
        # 将新加的列重命名，规则为原列名_值名
        t = t.rename(columns=lambda x: col+'_'+str(x))
        # 将新加的列放入x中，axis=1
        x = pd.concat((x, t), axis=1)
    # 取出x前10条看看
    print(x.head(10))


    # 系统对accept自动编码，便于之后作为标签使用
    y = np.array(pd.Categorical(data['accept']).codes)
    # 使用train_test_split方法切分开测试集与训练集，size表示切分的比例
    x, x_test, y, y_test = train_test_split(x, y, train_size=0.7,test_size=0.3)
    # 使用逻辑回归算法,cv是使用几折交叉验证，Cs是C的个数,C是正则惩罚项的系数倒数，C越小，惩罚越重,np.logspace是划分等比数列基数是10,起始指数为-3,终止为4,
    clf = LogisticRegressionCV(Cs=np.logspace(-3, 4, 8), cv=5)
    # 使用算法对测试集与训练集进行训练与测试
    clf.fit(x, y)
    # 输出C看一下
    print(clf.C_)

    # y_hat 为训练集算法后输出结果
    y_hat = clf.predict(x)
    # 输出训练精确度，metrics.accuracy_score，其中第一个参数为真实值，第二个参数为预测值
    print('训练集精确度：', metrics.accuracy_score(y, y_hat))
    # y_test_hat 为测试集算法后输出结果
    y_test_hat = clf.predict(x_test)
    # 同理，输出测试集精确度
    print('测试集精确度：', metrics.accuracy_score(y_test, y_test_hat))

    # 将y值的排重个数赋给n_class，np.unique为去除数组中的重复数字，然后排序输出
    n_class = len(np.unique(y))

    #判断y值个数是否大于2，若是，则为多分类问题
    if n_class > 2:
        #对测试集进行one_hot编码，类型个数取n_class，np.arange(a)输出从0到a
        y_test_one_hot = label_binarize(y_test, classes=np.arange(n_class))
        #predict_proba是指返回预测属于某个标签的概率，并将之赋给y_test_one_hot_hat
        y_test_one_hot_hat = clf.predict_proba(x_test)
        #numpy.ravel()将数据降为1维，metrics.roc_curve用来画ROC曲线,第一个参数为真实标签，第二个参数为预测结果标签，FPR为假正例率，TPR为真正例率
        fpr, tpr,_ = metrics.roc_curve(y_test_one_hot.ravel(), y_test_one_hot_hat.ravel())
        #metrics.auc用来求AUC的函数，参数分别为真正例率和假正例率，直接使用metrics.auc为手动计算
        print('Micro AUC:\t', metrics.auc(fpr, tpr))
        #计算得分曲线下的面积，第一个参数为真实标签，第二个参数为预测结果标签，参数必须为二值，average采用'micro'，使用函数直接计算
        auc = metrics.roc_auc_score(y_test_one_hot, y_test_one_hot_hat, average='micro')
        #输出系统AUC的值
        print('Micro AUC(System):\t', auc)
        #同理，计算得分曲线下的面积，第一个参数为真实标签，第二个参数为预测结果标签，参数必须为二值，average采用'macro'
        auc = metrics.roc_auc_score(y_test_one_hot, y_test_one_hot_hat, average='macro')
        #输出AUC的值
        print('Macro AUC:\t', auc)

    #否则是二分类问题
    else:
        #二分类问题可以直接进行AUC求解
        fpr, tpr, _ = metrics.roc_curve(y_test.ravel(), y_test_hat.ravel())
        # 输出AUC
        print('AUC:\t', metrics.auc(fpr, tpr))
        #roc_auc_score计算AUC的值
        auc = metrics.roc_auc_score(y_test, y_test_hat)
        #输出
        print('AUC(System):\t', auc)



    #以下为画图的相关设置

    #设置字体为微软雅黑
    mpl.rcParams['font.sans-serif'] = 'SimHei'
    #调整轴上的负号正常显示
    mpl.rcParams['axes.unicode_minus'] = False
    #画图的大小，figsize为图像宽和高，dpi为像素，facecolor为背景颜色
    plt.figure(figsize=(8, 7), dpi=80, facecolor='w')
    #plot为画图函数，有很多参数，用来调整图像
    plt.plot(fpr, tpr, 'b', lw=2, label='AUC=%.4f' % auc)
    #设置图例，参数loc为图例位置
    plt.legend(loc='lower right')
    #x坐标轴显示范围
    plt.xlim((-0.01, 1.02))
    #y坐标轴显示范围
    plt.ylim((-0.01, 1.02))
    #设置x轴刻度
    plt.xticks(np.arange(0, 1.1, 0.1))
    #设置y轴刻度
    plt.yticks(np.arange(0, 1.1, 0.1))
    #设置x轴标签，fontsize为字体大小
    plt.xlabel('False Positive Rate', fontsize=14)
    #设置y轴标签
    plt.ylabel('True Positive Rate', fontsize=14)
    #设置图像网格线，b为是否显示，ls为线段风格
    plt.grid(b=True,ls = ':')
    #设置图像标题
    plt.title('ROC曲线和AUC', fontsize=18)
    #图像显示
    plt.show()
