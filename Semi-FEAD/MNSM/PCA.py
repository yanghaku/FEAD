import numpy as np


# PCA降维
# 讲解链接: https://zhuanlan.zhihu.com/p/77151308
class PCA:
    # 初始化
    def __init__(self):
        # loading 矩阵
        self.P = None
        self.v = None
        self.x_mean = None  # 训练集的均值
        self.sumT = None  # 训练数据的分数矩阵的协方差矩阵的逆
        self.n_components = 0
        self.fit_Q = None  # 训练的数据集的Q统计量
        self.fit_D = None  # 训练的数据集的D统计量
        self.ucl_Q = None  # 训练的数据集的Q统计量的上99%分位数
        self.ucl_D = None  # 训练的数据集的D统计量的上99%分位数
        self.fit_anomaly_score = None  # 训练集的异常分数

    # X 是n*m的矩阵, n行m列, 每行是一个样本, 每个样本的维度为m
    # 将样本的维度从m降为n_components
    def fit(self, X, n_components, pca_w):
        # 均值化零, 并且乘缩放参数
        self.x_mean = X.mean(0)
        X = (X - self.x_mean) * pca_w

        self.n_components = n_components
        # 求协方差矩阵
        cov = np.dot(X.T, X) / (X.shape[0] - 1)

        # 求特征值和特征向量，特征向量是按列放的，即一列代表一个特征向量
        w, self.v = np.linalg.eig(np.mat(cov))
        if w.dtype is np.dtype('complex128'):
            w = w.real
            self.v = self.v.real

        # 通过特征值获取前N大的特征向量的id
        idx = np.argsort(-w)[:self.n_components]

        # 通过特征向量获取转化矩阵
        self.P = (self.v.T[idx, :]).T

        # 降维后的矩阵
        score = np.dot(X, self.P)

        # 求score的协方差
        score_mean = np.mean(score, axis=0)
        d = score - score_mean
        cov = np.dot(d.T, d) / (score.shape[0] - 1)
        # 求协方差的逆
        self.sumT = np.linalg.inv(np.mat(cov))

        # 计算校准数据的Q统计量和D统计量
        self.fit_D, self.fit_Q = self.Dst_Qst(X, None, score)

        self.ucl_D = np.quantile(self.fit_D, 0.99)
        self.ucl_Q = np.quantile(self.fit_Q, 0.99)

        self.fit_anomaly_score = self.anomaly_score(self.fit_D, self.fit_Q)
        # 返回分数矩阵
        return score

    # 计算Q统计量和D统计量
    def Dst_Qst(self, X, pca_w=None, t=None):
        if len(X.shape) == 1:  # 如果X是一个特征向量
            X = X.reshape((1, X.shape[0]))

        # 如果w不为空,就要均值化0
        if pca_w is not None:
            X = (X - self.x_mean) * pca_w

        if t is None:
            t = np.dot(X, self.P)
        e = X - np.dot(t, self.P.T)

        n = X.shape[0]
        Dst = np.empty(n)
        Qst = np.empty(n)
        for i in range(n):
            Dst[i] = np.dot(np.dot(t[i], self.sumT), t[i].T)[0, 0]
            Qst[i] = np.dot(e[i], e[i].T)[0, 0]
        return Dst, Qst

    def anomaly_score(self, Dst, Qst):
        M = self.P.shape[0]
        A = self.P.shape[1]
        scores = np.empty(Dst.shape[0])
        for i in range(Dst.shape[0]):
            msnm = (Dst[i] * A / (M * self.ucl_D)) + ((M - A) * Qst[i] / (M * self.ucl_Q))
            scores[i] = msnm
        return scores
