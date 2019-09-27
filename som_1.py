import numpy as np
import pylab as pl


class SOM(object):
    def __init__(self, X, output, iteration, batch_size):
        """
        :param X:  input 数据; 形状是 N*D， 输入样本有 N 个,每个 D 维
        :param output: (n,m) 一个元组，为输出层的形状是一个 n*m 的二维矩阵
        :param iteration: 迭代次数
        :param batch_size: 每次迭代时的样本数量
        初始化一个权值矩阵 W，形状为 D*(n*m)，即有 n*m 权值向量，每个 D 维; 猜测:第 i 行是第 i 个神经元在空间中的位置.
        """
        self.X = X
        self.output = output
        self.iteration = iteration
        self.batch_size = batch_size
        self.W = np.matrix(np.random.rand(output[0] * output[1], X.shape[1]))  * 10  # 每一列表示一个神经元的位置, 这个值得好好考虑
        # print(self.W.shape)

    def GetN(self, t):
        """ 优胜邻域
        :param t:时间t, 这里用迭代次数来表示时间
        :return: 返回一个整数，表示拓扑距离，时间越大，拓扑邻域越小
        """
        a = min(self.output)            # 神经元的行列的最小值
        return int(a - float(a) * t / self.iteration)   # 取值 0,1,2

    def Geteta(self, t, n):
        """
        :param t: 时间t, 这里用迭代次数来表示时间
        :param n: 拓扑距离
        :return: 返回学习率 eta，
        """
        eta = np.power(np.e, -n) / (t + 2)
        # print(eta) 
        return eta

    def updata_W(self, X, t, winner):
        # X 是数据点, t 是时间(迭代次数), winner: 最近的神经元
        N = self.GetN(t)                # N 是随着 t(迭代次数) 递减的, 是需要修改的神经元的最远距离.
        print(N) # 2 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  0

        for idx, item in enumerate(winner):  # 对每个样本(的最近神经元)循环
            to_update = self.getneighbor(item, N)   # item 是当前神经元; 获取当前神经元邻域半径N之内的神经元
            for j in range(N + 1):      # 对于不同的距离 0 -> N.
                e = self.Geteta(t, j)   # t 是迭代次数(时间), j 是距离
                for w in to_update[j]:
                    self.W[w, :] = np.add(self.W[w, :], e * (X[idx, :] - self.W[w, :]))

    def getneighbor(self, index, N):
        """
        :param index:获胜神经元的下标
        :param N: 邻域半径
        :return ans: 返回一个集合列表，分别是不同邻域半径内需要更新的神经元坐标
        """
        a, b = self.output  # 神经元大小.
        length = a * b

        def distence(index1, index2):
            i1_a, i1_b = index1 // a, index1 % b        # 整除和取余
            i2_a, i2_b = index2 // a, index2 % b
            return np.abs(i1_a - i2_a), np.abs(i1_b - i2_b)

        ans = [set() for i in range(N + 1)]
        for i in range(length):
            dist_a, dist_b = distence(i, index)
            if dist_a <= N and dist_b <= N: ans[max(dist_a, dist_b)].add(i)
        return ans
    
    def dist(self, x, w):
        # 数据需要按行输入, 每行表示一个向量. 这里为了简单, w是转置过的, 使用要小心.
        X = x.A         #  临时变量
        X_train = w.A
        dist = np.reshape(np.sum(X**2,axis=1),(X.shape[0],1))+ np.sum(X_train**2,axis=1)-2*X.dot(X_train.T)
        return dist

    def train(self):
        """
        train_Y: 训练样本与形状为batch_size*(n*m)
        winner:  一个一维向量，batch_size个获胜神经元的下标
        return:  返回值是调整后的 W
        """
        count = 0               # 迭代次数, 并不是拓扑距离...
        while self.iteration > count:
            train_X = self.X[np.random.choice(self.X.shape[0], self.batch_size)]    # 可重复抽样

            # normal_W(self.W)    # 归一化每个神经元的坐标
            # normal_X(train_X)   # 归一化每个样本的坐标

            # train_Y = train_X.dot(self.W)       # 直接做内积, 来判断距离
            train_Y = self.dist(train_X, self.W)     # train_Y 存放各样本与神经元的距离.

            # winner = np.argmax(train_Y, axis=1).tolist()        # 求最小的距离.
            winner = np.argmin(train_Y, axis=1).tolist()        # 求最小的距离.
            self.updata_W(train_X, count, winner)   # 这与学习率的计算, 有出入.
            count += 1
        return self.W

    def train_result(self):
        # normal_X(self.X)
        # train_Y = self.X.dot(self.W)
        train_Y = self.dist(self.X, self.W)     # train_Y 存放各样本与神经元的距离.
        clusters = np.argmin(train_Y, axis=1).tolist()        # 求最小的距离.
        print(clusters)       # 分类结果
        return clusters, self.W


def normal_X(X):
    """
    :param X:二维矩阵，N*D，N个D维的数据
    :return: 将 X 按行归一化的结果
    """
    N, D = X.shape
    for i in range(N):
        temp = np.sum(np.multiply(X[i], X[i]))
        X[i] /= np.sqrt(temp)
    return X


def normal_W(W):
    """
    :param W:二维矩阵，D*(n*m)，D个n*m维的数据, W 的每一列都是一个神经元的坐标. 
    :return: 将 W 按列归一化的结果
    """
    for i in range(W.shape[1]):
        temp = np.sum(np.multiply(W[:, i], W[:, i]))
        W[:, i] /= np.sqrt(temp)
    return W


# 画图
def draw(C, W=None):
    colValue = ['y', 'g', 'b', 'c', 'k', 'm', 'peru', 'darkorchid']
    for i in range(len(C)):
        coo_X = []  # x坐标列表
        coo_Y = []  # y坐标列表
        for j in range(len(C[i])):
            coo_X.append(C[i][j][0])
            coo_Y.append(C[i][j][1])
        pl.scatter(coo_X, coo_Y, marker='o', color=colValue[i % len(colValue)], label=i)

    pl.legend(loc='upper right')


    pl.plot(W[:,0], W[:,1], "ro", marker='x', markersize="20")

    pl.show()


if __name__ == '__main__':

    # 自己生成数据
    n = 50
    p = 2
    if p == 2:
        data = np.matrix(np.r_[np.random.normal(size=[n,p]) + [20,0], np.random.normal(size=[n,p]) + [0,20], np.random.normal(size=[n,p]) + [0,0], np.random.normal(size=[n,p]) + [15,15], np.random.normal(size=[n,p]) + [4,10], np.random.normal(size=[n,p]) + [10,4]])
    elif p == 3:
        data = np.matrix(np.r_[np.random.normal(size=[20,3]) + [20,0,0], np.random.normal(size=[20,3]) + [0,0,20], np.random.normal(size=[20,3]) + [0,0,0], np.random.normal(size=[20,3]) + [15,15,15], np.random.normal(size=[20,3]) + [4,10,5], np.random.normal(size=[20,3]) + [10,4,10]])
    dataset = data  # numpy 矩阵
    dataset_old = dataset.copy()

    # SOM
    som = SOM(dataset, (2, 4), 50, 100)   # X, output, iteration, batch_size
    som.train()

    # Reuslt
    classify = {}
    res = som.train_result()

    for i, win in enumerate(res[0]):
        if not classify.get(win):
            classify.setdefault(win, [i])
        else:
            classify[win].append(i)
    res_org = []   # 未归一化的数据分类结果
    for i in classify.values():
        res_org.append(dataset_old[i].tolist())

    draw(res_org, res[1])
    # draw(D)
