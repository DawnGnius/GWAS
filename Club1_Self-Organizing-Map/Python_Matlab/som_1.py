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
        # self.W = np.matrix(np.random.rand(output[0] * output[1], X.shape[1]))  * 10  # 每一列表示一个神经元的位置, 这个值得好好考虑
        self.W = np.matrix(np.zeros([output[0] * output[1], X.shape[1]]))
        self.count = 0
        self.GetGrid()
        # print(self.W.shape)

    def GetGrid(self):
        # 单位网格
        for idx in range(self.output[0]*self.output[1]):
            # 获取坐标
            r_idx = idx // self.output[1] + 1
            c_idx = idx - (r_idx-1) * self.output[1] + 1
            x_idx = (c_idx - 1) / (self.output[1] - 1)
            y_idx = (self.output[0] - r_idx) / (self.output[0] - 1)
            self.W[idx, 0], self.W[idx, 1] = x_idx-0.15, y_idx-0.15

        # 数据的极差
        ed = float(max(max(self.X[:,0])-min(self.X[:,0]), max(self.X[:,1])-min(self.X[:,1])))
        self.W = self.W * ed * 11 / 10

        file_name = './fig4/fig0.png'
        self.draw(W=self.W, data=self.X, file_name=file_name)

    def GetN(self, t):
        """ 优胜邻域
        :param t:时间t, 这里用迭代次数来表示时间
        :return: 返回一个整数，表示拓扑距离，时间越大，拓扑邻域越小
        """
        a = self.output[0] + self.output[1] - 2             # 神经元之间的最远距离
        if t < 0:
            return a-1
        else:
            # return int(a - float(a) * t / self.iteration)   # 取值
            return int(np.power(1.115, -t) * a)

    def Geteta(self, t, dist):
        """
        :param t: 时间t, 这里用迭代次数来表示时间
        :param n: 拓扑距离
        :return: 返回学习率 eta，
        """
        eta = np.power(np.e, -dist) / (t + 40)
        # print(eta)
        return eta

    def updata_W(self, X, t, winner):
        # X 是数据点, t 是时间(迭代次数), winner: 最近的神经元
        Neighborhood = self.GetN(t)                 # Neighborhood 是随着 t(迭代次数) 递减的, 是需要修改的神经元的最远距离.
        self.cur_neighbor = Neighborhood            # 用于绘图
        # print(Neighborhood)

        for idx, item in enumerate(winner):         # 对每个样本(的最近神经元)循环
            to_update = self.getneighbor(item, Neighborhood)   # item 是当前神经元; 获取当前神经元邻域半径N之内的神经元
            for j in range(Neighborhood + 1):       # 对于不同的距离 0 -> Neighborhood.
                eta= self.Geteta(t, j)              # t 是迭代次数(时间), j 是距离
                self.cur_eta = eta                  # 用于绘图
                for w in to_update[j]:
                    self.W[w, :] = np.add(self.W[w, :], eta * (X[idx, :] - self.W[w, :]))

    def getneighbor(self, index, Neighborhood):
        """
        :param index:获胜神经元的下标
        :param Neighborhood: 邻域半径
        :return ans: 返回一个集合列表，分别是不同邻域半径内需要更新的神经元坐标
        """
        a, b = self.output  # 神经元的个数.
        length = a * b

        ans = [set() for i in range(Neighborhood + 1)]
        for i in range(length):
            dist = self.Manhattan_dist(i, index)
            if dist <= Neighborhood: 
                ans[dist].add(i)
        return ans
    
    def Manhattan_dist(self, index1, index2):
        b = self.output[1]
        r_index1, r_index2 = index1 // b, index2 // b
        c_index1, c_index2 = index1-r_index1*b+1, index2-r_index2*b+1
        return np.abs(r_index1 - r_index2) + np.abs(c_index1 - c_index2) # 曼哈顿距离

    def Euclid_dist(self, x, w):
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
        self.count = 0               # 迭代次数, 并不是拓扑距离...
        while self.iteration > self.count:
            train_X = self.X[np.random.choice(self.X.shape[0], self.batch_size)]    # 可重复抽样

            train_Y = self.Euclid_dist(train_X, self.W)     # train_Y 存放各样本与神经元的距离.

            winner = np.argmin(train_Y, axis=1).tolist()        # 求最小的距离.
            self.updata_W(train_X, self.count, winner)   # 这与学习率的计算, 有出入.
            self.count += 1

            # save learning result
            file_name = './fig4/fig'+str(self.count)+'.png'
            self.draw(W=self.W, data=self.X, file_name=file_name)

        return self.W

    def train_result(self):
        train_Y = self.Euclid_dist(self.X, self.W)              # train_Y 存放各样本与神经元的距离.
        clusters = np.argmin(train_Y, axis=1).tolist()          # 求最小的距离.
        # print(clusters)                                         # 分类结果
        return clusters, self.W

    def draw(self, W, data=[], C=[], file_name=1):
        pl.figure(figsize=(14, 6))
        colValue = ['y', 'g', 'b', 'c', 'k', 'm', 'peru', 'darkorchid']

        # 聚类图像
        pl.subplot(1,2,1)
        # 普通的样本
        if len(data) != 0:
            pl.plot(data[:,0], data[:,1], 'ko', markersize="5")

        # 分类的样本
        if len(C) != 0:
            for i in range(len(C)):
                coo_X = []  # x坐标列表
                coo_Y = []  # y坐标列表
                for j in range(len(C[i])):
                    coo_X.append(C[i][j][0])
                    coo_Y.append(C[i][j][1])
                pl.scatter(coo_X, coo_Y, marker='o', color=colValue[i % len(colValue)], label=i)

            pl.legend(loc='upper right')

        # 神经元
        pl.plot(W[:,0], W[:,1], "ro", marker='o', markersize="7")
        for index in range(self.output[0]*self.output[1]):
            for i in range(self.output[0]*self.output[1]):
                dist = self.Manhattan_dist(i, index)
                if dist == 1: 
                    pl.plot(W[[i,index],0], W[[i,index],1], "r")
        
        pl.xlabel('X')
        pl.ylabel('Y')
        pl.title('SOM')
        pl.xlim(-5, 25)
        pl.ylim(-5, 25)

        # 邻域半径图像
        pl.subplot(2,2,2)
        pl.title('Radius of Neighborhood')
        pl.ylabel('Manhattan Distance')
        x = np.linspace(0, self.iteration, self.iteration+1)
        y = np.power(1.115, -x) * 4
        pl.plot(x,y)
        font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 18}
        pl.text(20,3.5,r'$D = 4 \times 1.115^{-t}$', font1)
        if file_name!=1:
            pl.plot(self.count,  np.power(1.115, -self.count) * 4, 'go', markersize="12")

        # 绘制学习率图像
        pl.subplot(2,2,4)
        pl.title('Learning Rate')
        pl.xlabel('Iteration')
        pl.ylabel('Rate')
        x = np.linspace(0, self.iteration, self.iteration+1)
        y = np.power(np.e, 0) / (x + 40)
        pl.plot(x,y)
        pl.text(20,0.024,'Batch Size=100, D=0', font1)
        pl.text(20,0.022, r'$R = e^{-D} / (t + 40)$', font1)
        if file_name!=1:
            pl.plot(self.count, np.power(np.e, 0) / (self.count + 40), 'go', markersize="12")
        # 保存图片
        if file_name==1:
            pl.savefig('./fig4/fig36.png', dpi=1200)
            pl.show()
            
        else:
            pl.savefig(file_name, dpi=1200)
            pl.close()


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
    som = SOM(dataset, (3, 3), 35, 100)     # X, output, iteration, batch_size
    som.train()

    # Reuslt
    classify = {}
    res = som.train_result()                # 返回聚类结果以及神经元

    for i, win in enumerate(res[0]):
        if not classify.get(win):
            classify.setdefault(win, [i])
        else:
            classify[win].append(i)
    res_org = []   # 未归一化的数据分类结果
    for i in classify.values():
        res_org.append(dataset_old[i].tolist())

    som.draw(C=res_org, W=res[1])
