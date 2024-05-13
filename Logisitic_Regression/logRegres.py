import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
matplotlib.use('TkAgg')


def load_Dataset():
    #load_Dataset是为了将testSet.txt文本中的数据提取出来，并进行预处理
    data_Mat = []
    label_Mat = []

    with open('Dataset/testSet.txt', 'r') as f:
        for line in f.readlines():
            line_Arr = line.strip().split()
            data_Mat.append([1.0, float(line_Arr[0]), float(line_Arr[1])])
            #将数据集中第一二列中的特征值放在data_Mat中，每一行都添加参数1.0，是将w0x0设置为常数w0
            label_Mat.append(int(line_Arr[2]))  #将数据集中第一二列中的特征值放在label_Mat中
    return data_Mat, label_Mat


def sigmoid(z):
    #sigmodi函数 将得到的值收缩在0~1之间，方便与实际标签计算
    return 1.0 / (1 + np.exp(-z))


def grad_ascent(data_mat_in, class_labels):
    #梯度上升法
    data_matrix = np.mat(data_mat_in)
    label_mat = np.mat(class_labels).transpose()        #将得到的特征和标签转换为Numpy矩阵数据类型

    m, n = np.shape(data_matrix)   #获得特征矩阵的形状，m(m=100)行，n(n=3)列
    alpha = 0.001                  #更新梯度的步长
    max_cycles = 500               #迭代次数
    weights = np.ones((n, 1))         #w0,w1,w2初始赋值为1.0
    for k in range(max_cycles):
        h = sigmoid(data_matrix * weights)  #得到特征x0,x1,x2和参数w0,w1,w2相乘后，用sigmoid处理后的值
        error = (label_mat - h) #计算label矩阵和预测值矩阵的差
        weights = weights + alpha * data_matrix.transpose() * error
        #用得到的error和特征矩阵和alpha参数相乘结果作为w0，w1，w2的更新幅度
    return weights


def plot_best_fit(weights):

    data_mat, label_mat = load_Dataset()
    data_arr = np.array(data_mat)
    n = np.shape(data_arr)[0]

    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(label_mat[i]) == 1:
            xcord1.append(data_arr[i, 1])
            ycord1.append(data_arr[i, 2])
        else:
            xcord2.append(data_arr[i, 1])
            ycord2.append(data_arr[i, 2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')

    # 最佳拟合直线
    x = np.arange(-3.0, 3.0, 0.1)
    print(weights)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def stoc_grad_ascent0(data_matrix, class_labels):
    #随机梯度上升法
    m, n = np.shape(data_matrix)
    alpha = 0.01
    weights = np.ones(n)
    #用数据集的每个样本单独更新一次weights，总共更新100次
    for i in range(m):
        h = sigmoid(np.sum(data_matrix[i] * weights)) #每次取出一个样本计算
        error = class_labels[i] - h
        weights = weights + alpha * error * np.array(data_matrix[i])
    return weights

def stoc_grad_ascent1(data_matrix, class_labels, num_iter=150):
    m, n = np.shape(data_matrix)
    weights = np.ones(n)
    for j in range(num_iter):
        data_index = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01 #每次迭代更新alpha
            rand_index = int(random.uniform(0, len(data_index))) # 随机选择更新，减少周期性的波动
            h = sigmoid(np.sum(data_matrix[rand_index] * weights))
            error = class_labels[rand_index] - h
            weights = weights + alpha * error * np.array(data_matrix[rand_index])
            del(data_index[rand_index]) #删除已经使用过的样本
    return weights


def classify_vector(in_x, weights):
    prob = sigmoid(np.sum(in_x * weights))

    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colic_test():
    fr_train = open('Dataset/horseColicTraining.txt')
    fr_test = open('Dataset/horseColicTest.txt')
    training_set = []
    training_labels = []

    for line in fr_train.readlines():
        curr_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(curr_line[i]))        #将训练集的所有特征值保存在training_set中
        training_set.append(line_arr)                   #最后一列的标签保存在training_labels
        training_labels.append(float(curr_line[21]))

    train_weights = stoc_grad_ascent1(np.array(training_set), training_labels, 500)
    #调用随机上升梯度法，根据训练集得到各个参数weights的值

    error_count = 0
    num_test_vec = 0.0
    for line in fr_test.readlines():
        num_test_vec += 1.0
        curr_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(curr_line[i]))    #取出测试集的所有信息保存在line_arr

        if int(classify_vector(np.array(line_arr), train_weights)) != int(curr_line[21]):
            #使用得到的参数weights与测试样本的特征相乘得到预测结果，预测结果和实际标签相比较
            error_count += 1    #统计预测错误数目

    error_rate = float(error_count) / num_test_vec #计算错误率
    print(f"the error rate of this test is {error_rate}")
    return error_rate


def multi_test():
    num_tests = 10
    error_sum = 0.0
    for k in range(num_tests):
        error_sum += colic_test()
    #平均num_tests次预测结果的错误率
    print(f"after {num_tests} iterations the average error rate is {error_sum/float(num_tests)}")


if __name__ == "__main__":
    # 1. 测试
    data_arr, label_mat = load_Dataset()
    result = stoc_grad_ascent1(data_arr, label_mat)
    print(result)
    plot_best_fit(result)

    # 2. 从疝气病预测病马的死亡率
    multi_test()
