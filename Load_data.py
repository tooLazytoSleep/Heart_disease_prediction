import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
#txt文件处理
def file_processed():
    file_1 = open('processed.txt')
    r_file = file_1.read()
    file_2 = open('a_processed.txt','w')
    file_2.write(r_file.replace(',',' '))
    file_1.close()
    file_2.close()
def sub_1(x):
    if x== 2:
        x=1
    elif x == 3:
        x=1
    elif x ==4:
        x=1
    return x
def sub_2(x):
    if x== 6:
        x=1
    elif x == 3:
        x=0
    elif x == 7:
        x=2
    return x
def Max_num(x):
    return max(x)
def Min_num(x):
    return min(x)
#数据预处理
def load_data(file):
    file_processed()
    features = np.loadtxt(file, dtype={'names': ('age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slop', 'ca','thal', 'status'),'formats': ('f4', 'f4', 'f4', 'f4', 'f4', 'f4','f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4')})
    df = pd.DataFrame(features, columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slop', 'ca','thal', 'status'])
    #调整数据
    df['cp'] = df['cp'].map(lambda x: x-1)
    df['slop'] = df['slop'].map(lambda x: x - 1)
    df['status'] = df['status'].map(lambda x: sub_1(x))
    df['thal'] = df['thal'].map(lambda x: sub_2(x))
    #归一化处理
    df['age'] = df['age'].map(lambda x: (x - Min_num(df['age'])) / (Max_num(df['age']) - Min_num(df['age'])))
    df['sex'] = df['sex'].map(lambda x: (x - Min_num(df['sex'])) / (Max_num(df['sex']) - Min_num(df['sex'])))
    df['cp'] = df['cp'].map(lambda x: (x - Min_num(df['cp'])) / (Max_num(df['cp']) - Min_num(df['cp'])))
    df['trestbps'] = df['trestbps'].map(lambda x: (x - Min_num(df['trestbps'])) / (Max_num(df['trestbps']) - Min_num(df['trestbps'])))
    df['chol'] = df['chol'].map(lambda x: (x - Min_num(df['chol'])) / (Max_num(df['chol']) - Min_num(df['chol'])))
    df['fbs'] = df['fbs'].map(lambda x: (x - Min_num(df['fbs'])) / (Max_num(df['fbs']) - Min_num(df['fbs'])))
    df['restecg'] = df['restecg'].map(lambda x: (x - Min_num(df['restecg'])) / (Max_num(df['restecg']) - Min_num(df['restecg'])))
    df['thalach'] = df['thalach'].map(lambda x: (x - Min_num(df['thalach'])) / (Max_num(df['thalach']) - Min_num(df['thalach'])))
    df['exang'] = df['exang'].map(lambda x: (x - Min_num(df['exang'])) / (Max_num(df['exang']) - Min_num(df['exang'])))
    df['oldpeak'] = df['oldpeak'].map(lambda x: (x - Min_num(df['oldpeak'])) / (Max_num(df['oldpeak']) - Min_num(df['oldpeak'])))
    df['slop'] = df['slop'].map(lambda x: (x - Min_num(df['slop'])) / (Max_num(df['slop']) - Min_num(df['slop'])))
    df['ca'] = df['ca'].map(lambda x: (x - Min_num(df['ca'])) / (Max_num(df['ca']) - Min_num(df['ca'])))
    df['thal'] = df['thal'].map(lambda x: (x - Min_num(df['thal'])) / (Max_num(df['thal']) - Min_num(df['thal'])))
    df['status'] = df['status'].map(lambda x: (x - Min_num(df['status'])) / (Max_num(df['status']) - Min_num(df['status'])))
    return (df.values)
#print(load_data('a_processed.txt'))
#拆分数据（训练集与测试集）
def split_data(file):
    data = load_data(file)
    X = data[:,:-1]
    Y = data[:,-1]
    X_train,X_test,y_train,y_test = train_test_split(X, Y,test_size=0.3)

    return (X_train, X_test, y_train, y_test)
X, X_test, y, y_test = split_data('a_processed.txt')
#逻辑回归二分类模型
model = LogisticRegression(solver='liblinear')
model.fit(X,y)
model.score(X,y)
weights = model.coef_
intercept = model.intercept_
predicted = model.predict(X_test)
#伯努利朴素贝叶斯模型
clf = BernoulliNB()
clf.fit(X, y)
BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
#多项式分布模型
mul = MultinomialNB().fit(X, y)
#print (clf.predict(X_test))
#预测模型
def prediction(pre, ac, right, wrong):
    for i in range(len(pre)):
        if pre[i] == ac[i]:
            right = right +1
        else:
            wrong = wrong +1
    rate = right / (right + wrong)
    return rate

print("利用逻辑回归模型计算准确率：", prediction(predicted, y_test, 0, 0))
print("利用伯努利朴素贝叶斯模型计算准确率：", prediction(clf.predict(X_test), y_test, 0, 0))
print("利用多项式模型计算准确率：", prediction(mul.predict(X_test), y_test, 0, 0))