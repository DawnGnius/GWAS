import random
import matplotlib.pyplot as plt
import numpy as np

def initOutputLayer(num_layer, num_dim):  # num_layer 为竞争层节点数目；num_dim 为每一个节点的维度
    # 返回值为神经元的位置列表
    layers = []
    random.seed()
    for i in range(num_layer):
        unit = []  # 每一个节点
        for j in range(num_dim):
            unit.append(round(random.random(),2) * 10)
        layers.append(unit)
    return layers 

def normalization(v):  # v为向量
    norm = np.linalg.norm(v, 2)  # 计算2范数
    v_new = []
    for i in range(len(v)):
        v_new.append(round(v[i]/norm,2))  # 保留2位小数
    # return v_new
    return v

def normalizationVList(X):  
    X_new = []
    for x in X:
        X_new.append(normalization(x))
    return X_new

def calSimilarity(x, y):  # 计算x,y两个向量的相似度, 欧氏距离
    if len(x) != len(y):
        raise "维度不一致！"
    c = 0
    for i in range(len(x)):
        c += pow((x[i] - y[i]), 2)
    return  np.sqrt(c)

def getWinner(x, layers):  # 找到 layers 里面与 x 最相似的节点
    # x = normalization(x)
    # layers = normalizationVList(layers)
    min_value = 100000  # 存储最短距离
    min_index = -1  # 存储跟x最相似节点的竞争层节点index
    for i in range(len(layers)):
        v = calSimilarity(x, layers[i])
        if v < min_value:
            min_value = v
            min_index = i
    return min_index  # 返回获胜节点index

def adjustWeight(w, x, alpha):  # w为要调整的权值向量；x为输入向量；alpha为学习率
    if len(w) != len(x):
        raise "w, x维度应该相等！"
    w_new = []
    for i in range(len(w)):
        w_new.append(w[i] + alpha*(x[i] - w[i]))
    return w_new

def createData(num, dim):  # 数据组数与数据维度
    data = []
    for i in range(num):
        pair = []
        for j in range(dim):
            pair.append(random.random())
        data.append(pair)
    return data

def createData2(n, p):
    n = int(n)
    p = int(p)
    if p == 2:
        data = np.matrix(np.r_[np.random.normal(size=[n,p]) + [20,0], np.random.normal(size=[n,p]) + [0,20], np.random.normal(size=[n,p]) + [0,0], np.random.normal(size=[n,p]) + [15,15], np.random.normal(size=[n,p]) + [4,10], np.random.normal(size=[n,p]) + [10,4]])
    elif p==3:
        data = np.matrix(np.r_[np.random.normal(size=[n,p]) + [20,0,0], np.random.normal(size=[n,p]) + [0,0,20], np.random.normal(size=[n,p]) + [0,0,0], np.random.normal(size=[n,p]) + [15,15,15], np.random.normal(size=[n,p]) + [4,10,5], np.random.normal(size=[n,p]) + [10,4,10]])
    
    # return np.array(normalizationVList(data.A))
    return data.A   # 转换成 np.array
    
# 参数设置
train_times = 500       # 训练次数
total_times = train_times
data_dim = 2            # 数据维度
train_num = 160
test_num = 40
learn_rate = 0.5        # 学习率初值

# 生成数据
random.seed()
# 生成训练数据
train_X = createData2(train_num/4, data_dim)
# 生成测试数据
test_X = createData2(test_num/2, data_dim)
# print(test_X)
# plt.plot(test_X[:,0], test_X[:,1],"o")
# plt.show()

# 更新数据个数
train_num = train_X.shape[0]
test_num = test_X.shape[0]

# 初始化 m 个类
m = 6  # m个类别
layers = initOutputLayer(m, data_dim)
print("Original layers:", layers)

# 开始迭代训练
while train_times > 0:
    for i in range(train_num):
        # 权值归一化
        layers_norm = normalizationVList(layers)
        # 计算某一个x输入的竞争层胜利节点
        winner_index = getWinner(train_X[i], layers_norm)
        # 修正权值
        layers[winner_index] = adjustWeight(layers[winner_index], train_X[i], learn_rate*train_times/total_times)
    train_times -= 1
print("After train layers:", layers)

# 测试
for i in range(test_num):
    # 权值归一化
    layers_norm = normalizationVList(layers)
    # 计算某一个x输入的竞争层胜利节点
    winner_index = getWinner(test_X[i], layers_norm)
    # 画图
    colValue = ["gv", 'yo', 'go', 'bo', 'co', 'ko', 'mo']

    plt.plot(test_X[i, 0], test_X[i, 1], colValue[winner_index])
# plt.legend()

layers_a = np.array(layers)
plt.plot(layers_a[:,0], layers_a[:,1], "ro", marker='x', markersize="10")
plt.show()