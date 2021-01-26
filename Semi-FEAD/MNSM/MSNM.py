from PCA import PCA
from PLS import PLS
import numpy as np
from sklearn.metrics import roc_auc_score


class MSNM:
    # M是输入变量的维度
    def __init__(self, M):
        self.pca = PCA()
        # 传入pca的缩放参数
        self.pca_w = np.ones(M)
        self.M = M

    # 训练, 并且使用R2R优化, n_component是主成分分析降维的个数, K 是 R2R的计算次数, rc是设置的w的偏移率
    def train(self, label_train, train_labels, unlabel_train, n_component=50, K=10, rc=0.1, epoch=20):
        pls = PLS()
        S = np.empty((K, self.M))
        Y = np.empty(K)
        for e in range(epoch):
            for i in range(K):
                w = self.pca_w + rc * np.random.normal(0, 1, self.M)
                self.pca.fit(unlabel_train, n_component, w)
                Dst, Qst = self.pca.Dst_Qst(label_train, w)
                score = self.pca.anomaly_score(Dst, Qst)
                auc = roc_auc_score(train_labels, score)
                S[i, :] = w
                Y[i] = auc
            pls.fit(S, Y, 10)
            self.pca_w = self.pca_w + (3 * pls.B).reshape(self.M)

            print("epoch ", e + 1, "/", epoch)

    # 返回异常分数
    def test(self, X):
        # 计算异常分数
        Dst, Qst = self.pca.Dst_Qst(X, self.pca_w)
        scores = self.pca.anomaly_score(Dst, Qst)
        return scores

    # def predict(self, X):
    #     X = X - X.mean(0)
    #     pre = []
    #     for i in X:
    #         Dst, Qst = self.pca.Dst_Qst(i)
    #         if Dst > self.pca.ucl_D and Qst > self.pca.ucl_Q:
    #             pre.append(1)
    #         else:
    #             pre.append(0)
    #     return pre