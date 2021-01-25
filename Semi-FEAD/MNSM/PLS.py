import numpy as np


# 偏最小二乘法回归
class PLS:

    def __init__(self):
        # X , Y 的第一维
        self.N = 0
        # X 的 第二维, 也就是X的特征向量大小
        self.Mx = 0
        # Y 的 第二维, Y的特征向量大小
        self.My = 0
        # # 训练集 X 的 均值
        # self.X_mean = None
        # # 训练集 Y 的 均值
        # self.Y_mean = None
        # # 训练集 X 的 标准差
        # self.X_std = None
        # # 训练集 Y 的 标准差
        # self.Y_std = None
        # # X 的缩放参数
        # self.X_scale = None
        # # Y 的缩放参数
        # self.Y_scale = None
        self.P = None
        self.W = None
        self.Q = None
        self.U = None
        self.T = None
        self.b = None
        self.B = None

    # 使用PLS拟合当前的模型, 假设X,Y都已经标准化(均值为0)
    def fit(self, X, Y, d=None, max_iter=500, eps=1e-6):
        # 预处理, 把一维变成二维
        if len(X.shape) == 1:
            X = X.reshape((X.shape[0], 1))
        if len(Y.shape) == 1:
            Y = Y.reshape((Y.shape[0], 1))
        self.N = X.shape[0]  # X.shape[0]==Y.shape[0]
        self.Mx = X.shape[1]
        self.My = Y.shape[1]

        # self.X_mean = X.mean(0)
        # # ddof=1表示求方差除以N-1
        # self.X_std = X.std(0, ddof=1) + 1e-10
        # self.X_scale = 1.0 / self.X_std
        # self.Y_mean = Y.mean(0)
        # self.Y_std = Y.std(0, ddof=1) + 1e-10
        # self.Y_scale = 1.0 / self.Y_std
        #
        # # 将X, Y 标准化
        # Y_norm = (Y - self.Y_mean) * self.Y_scale
        # X_norm = (X - self.X_mean) * self.X_scale

        P = np.empty((self.Mx, d))
        W = np.empty((self.Mx, d))
        Q = np.empty((self.My, d))
        U = np.empty((self.N, d))
        T = np.empty((self.N, d))
        b = np.empty((d))

        # 计算d个主成分
        for it in range(d):
            # 随机选出一个变量特征来
            u = Y[:, np.random.randint(self.My)].reshape(self.N, 1)
            p = np.empty((self.Mx))
            q = np.empty((self.My))
            t = np.empty((self.N))

            # 迭代max_iter次(可能提前停止)
            for jt in range(max_iter):
                p = np.matmul(X.T, u)
                # 除以二范数
                p /= (np.linalg.norm(p, 2) + 1e-10)
                t = np.matmul(X, p)
                q = np.matmul(Y.T, t)
                # 除以二范数
                q /= (np.linalg.norm(q, 2) + 1e-10)
                u_new = np.matmul(Y, q)
                # 如果两次迭代的差值小于eps,就提前停止
                if np.linalg.norm(u - u_new, 2) <= eps:
                    u = u_new
                    break
                u = u_new

            W[:, it] = p.reshape((self.Mx))
            Q[:, it] = q.reshape((self.My))
            U[:, it] = u.reshape((self.N))
            T[:, it] = t.reshape((self.N))
            t_sum = np.sum(t * t) + 1e-10
            # 计算回归系数
            b[it] = np.sum(u * t) / t_sum

            P[:, it] = (np.matmul(np.transpose(X), t) / t_sum).reshape((self.Mx))

            # 更新残差
            X -= np.matmul(t, P[:, it].reshape(1, self.Mx))
            Y -= b[it] * np.matmul(t, q.T)

        self.P = P[:, 0:d]
        self.W = W[:, 0:d]
        self.Q = Q[:, 0:d]
        self.U = U[:, 0:d]
        self.T = T[:, 0:d]
        self.b = b[0:d]
        self.B = np.matmul(
            np.matmul(self.W, np.linalg.inv(np.matmul(self.P.T, self.W))),
            np.matmul(np.diag(self.b), self.Q.T)
        )

    # def predict(self, X):
    #     # X 是 1 * Mx的时候
    #     if len(X.shape) == 1:
    #         Y = self.Y_mean + np.matmul(((X - self.X_mean) * self.X_scale).T, self.B) * self.Y_std
    #     else:
    #         N_pred = X.shape[0]
    #         Y = np.empty((N_pred, self.My))
    #         for it in range(N_pred):
    #             Y[it, :] = self.Y_mean + np.matmul(((X[it, :] - self.X_mean) * self.X_scale).T, self.B) * self.Y_std
    #
    #     return Y
