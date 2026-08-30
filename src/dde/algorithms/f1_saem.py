# utilities/F_1_SAEM.py
import numpy as np

def F_1_SAEM(Xj, A1_sample_long, N, K1, C):
    """
    返回一个闭包 loglike(x)，对应 MATLAB: f = @loglike
    计算（固定 j 时）的负对数似然近似： sum((Xj - eta)^2)/C

    参数
    ----
    Xj : array-like, shape (N, C)
        观测/目标矩阵（固定第 j 个特征后的 N×C 面板）
    A1_sample_long : array-like, shape (N, K1+1, C)
        长格式的 A1 样本张量；第二维长度需为 K1+1（含截距列）
    N : int
        样本数
    K1 : int
        A1 的维度（不含截距）；因此 A1_sample_long 第二维应为 K1+1
    C : int
        随机近似/样本条数（与 Xj 的列数一致）

    返回
    ----
    loglike : callable
        接受 x 向量（长度 K1+1），返回标量损失 sum((Xj - eta)^2)/C
    """
    Xj = np.asarray(Xj, dtype=float)
    A1_sample_long = np.asarray(A1_sample_long, dtype=float)

    # 基本维度检查，提前暴露错误
    if Xj.shape != (N, C):
        raise ValueError(f"Xj shape {Xj.shape} != (N, C)=({N}, {C})")
    if A1_sample_long.shape != (N, K1 + 1, C):
        raise ValueError(f"A1_sample_long shape {A1_sample_long.shape} != (N, K1+1, C)=({N}, {K1+1}, {C})")

    def loglike(x):
        """
        x : array-like, shape (K1+1,)
            系数/参数向量（含截距）
        返回
        ----
        ll : float
            sum((Xj - eta)^2)/C
        """
        x = np.asarray(x, dtype=float)
        if x.shape != (K1 + 1,):
            raise ValueError(f"x shape {x.shape} != (K1+1,) = ({K1+1},)")

        # eta = reshape(sum(bsxfun(@times, A1_sample_long, reshape(x,[1 K1+1 1])), 2), [N C])
        # 对应：沿 axis=1（第二维，K1+1）求和；广播 x 到 (1, K1+1, 1)
        eta = np.sum(A1_sample_long * x.reshape(1, K1 + 1, 1), axis=1)  # (N, C)

        # ll = sum((Xj - eta).^2, 'all')/C;
        resid = Xj - eta
        ll = float(np.sum(resid * resid) / float(C))
        return ll

    return loglike

# 备注（若需切换到 Bernoulli-Logistic 似然，可将 loglike 中最后两行替换为）：
#   # -(sum(eta,2)' * Xj - sum( log(exp(eta)+1), 'all'))/C;
#   ll = float(-(np.sum(eta, axis=1).T @ np.sum(Xj, axis=1) - np.sum(np.log1p(np.exp(eta)))) / float(C))
