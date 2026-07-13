# utilities/F_2_SAEM.py
import numpy as np

def F_2_SAEM(A1_sample_k, A2_sample_long, N, K2, C):
    """
    返回一个闭包 loglike(x)，对应 MATLAB: f = @loglike
    计算 (固定 k 时) 的负对数似然（logistic 模型形式）

    参数
    ----
    A1_sample_k : array-like, shape (N, C)
        第 k 个 A1 变量的采样矩阵
    A2_sample_long : array-like, shape (N, K2+1, C)
        A2 的长格式样本张量，第二维长度为 K2+1（含截距列）
    N : int
        样本数
    K2 : int
        A2 的维度（不含截距）
    C : int
        随机近似条数（列数）

    返回
    ----
    loglike : callable
        接受 x 向量 (K2+1,)，返回负对数似然标量：
        -(sum(A1 * eta) - sum(log(1 + exp(eta)))) / C
    """
    A1_sample_k = np.asarray(A1_sample_k, dtype=float)
    A2_sample_long = np.asarray(A2_sample_long, dtype=float)

    # 检查维度一致性
    if A1_sample_k.shape != (N, C):
        raise ValueError(f"A1_sample_k shape {A1_sample_k.shape} != (N, C)=({N}, {C})")
    if A2_sample_long.shape != (N, K2 + 1, C):
        raise ValueError(f"A2_sample_long shape {A2_sample_long.shape} != (N, K2+1, C)=({N}, {K2+1}, {C})")

    def loglike(x):
        """
        x : array-like, shape (K2+1,)
            参数向量（含截距）
        返回
        ----
        ll : float
            负对数似然值
        """
        x = np.asarray(x, dtype=float)
        if x.shape != (K2 + 1,):
            raise ValueError(f"x shape {x.shape} != (K2+1,) = ({K2+1},)")

        # eta = reshape(sum(bsxfun(@times, A2_sample_long, reshape(x, [1 K2+1 1])), 2), [N C]);
        eta = np.sum(A2_sample_long * x.reshape(1, K2 + 1, 1), axis=1)  # (N, C)

        # ll = -(sum(A1_sample_k .* eta, 'all') - sum(log(1+exp(eta)), 'all'))/C;
        ll = - (np.sum(A1_sample_k * eta) - np.sum(np.log1p(np.exp(eta)))) / float(C)
        return float(ll)

    return loglike
