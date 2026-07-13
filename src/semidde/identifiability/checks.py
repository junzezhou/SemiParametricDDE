import numpy as np


# ─── 内部工具 ─────────────────────────────────────────────────────────────────

def _binary_matrix(K: int) -> np.ndarray:
    """生成 (2**K, K) 的二进制矩阵（高位在前）。"""
    n = 1 << K
    idx = np.arange(n, dtype=np.int32)
    return ((idx[:, None] >> np.arange(K - 1, -1, -1)) & 1).astype(np.float64)


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _compute_propA_top(prop: np.ndarray):
    """顶层独立 Bernoulli 的联合 pmf。"""
    K = len(prop)
    A = _binary_matrix(K)
    lp  = np.log(np.clip(prop,       1e-12, 1 - 1e-12))
    l1p = np.log(np.clip(1.0 - prop, 1e-12, 1 - 1e-12))
    prob = np.exp(A @ lp + (1.0 - A) @ l1p)
    return A, prob / prob.sum()


def _marginal_next_layer(EXP_L: np.ndarray, propA_U: np.ndarray,
                         batch_size: int = 512, eps: float = 1e-12) -> np.ndarray:
    """
    计算 propA_L = condPropA_L @ propA_U，分批处理避免构造大矩阵。
    内存 O(batch_size × nU)。
    """
    K_L, nU = EXP_L.shape
    nL    = 1 << K_L
    P     = np.clip(EXP_L, eps, 1.0 - eps)
    base  = np.log(1.0 - P).sum(axis=0)
    delta = np.log(P / (1.0 - P))
    shift = np.arange(K_L - 1, -1, -1)
    propA_L = np.empty(nL, dtype=np.float64)
    for start in range(0, nL, batch_size):
        end  = min(start + batch_size, nL)
        idx  = np.arange(start, end, dtype=np.int32)
        bits = ((idx[:, None] >> shift) & 1).astype(np.float64)
        propA_L[start:end] = np.exp(base + bits @ delta) @ propA_U
    return propA_L


def _compute_local_joint(dep_indices: list,
                         propA_ref: np.ndarray,
                         A_ref: np.ndarray,
                         B_sub_mid: np.ndarray,
                         intercept_mid: np.ndarray) -> np.ndarray:
    """
    计算 P(L^(mid)_{dep} = a)，通过对可枚举的 ref 层 pmf 做局部边际化。

    dep_indices : mid 层中关心的分量索引，len = d ≤ 4
    B_sub_mid   : (K_mid, K_ref) 稀疏，每行 ≤ 2 个非零列
    返回        : (2**d,)

    由于每个 mid 分量最多依赖 2 个 ref 分量，d ≤ 4 时需枚举的 ref 局部配置
    最多 2**8 = 256 个，计算开销极小。
    """
    
    d = len(dep_indices)
    if d == 0:
        return np.array([1.0])

    ref_cols = sorted(
        set(c for k in dep_indices for c in np.where(B_sub_mid[k] != 0)[0])
    )
    n_ref = len(ref_cols)

    # 边际化 propA_ref 到 ref_cols
    powers    = 1 << np.arange(n_ref - 1, -1, -1)
    sub_idx   = (A_ref[:, ref_cols].astype(int)) @ powers
    joint_ref = np.zeros(2**n_ref)
    np.add.at(joint_ref, sub_idx, propA_ref)

    # 各 dep 分量的条件均值（在每个 ref 局部配置下）
    b_cfgs    = _binary_matrix(n_ref)
    sigma_dep = np.zeros((d, 2**n_ref))
    for ki, k in enumerate(dep_indices):
        w = B_sub_mid[k][ref_cols]
        sigma_dep[ki] = _logistic(intercept_mid[k] + b_cfgs @ w)

    # 枚举 dep 配置，计算联合边际
    a_cfgs    = _binary_matrix(d)
    joint_dep = np.zeros(2**d)
    for ai in range(2**d):
        a  = a_cfgs[ai]
        lc = np.zeros(2**n_ref)
        for ki in range(d):
            sk = np.clip(sigma_dep[ki], 1e-12, 1 - 1e-12)
            lc += np.log(sk) if a[ki] == 1 else np.log(1.0 - sk)
        joint_dep[ai] = (joint_ref * np.exp(lc)).sum()
    return joint_dep


def _compute_moments_sparse(B_sub_cur: np.ndarray, intercept_cur: np.ndarray,
                             B_sub_mid: np.ndarray, intercept_mid: np.ndarray,
                             propA_ref: np.ndarray, A_ref: np.ndarray):
    """
    精确计算 cur 层的一/二阶矩，跨越无法枚举的 mid 层。

    利用两层稀疏性（每行 ≤ 2 个非零列）：
      cur 层第 j 个变量最多依赖 2 个 mid 分量，
      每个 mid 分量最多依赖 2 个 ref 分量，
      因此 j 间接依赖 ≤ 4 个 ref 分量，局部枚举仅需 2**4 = 16 个配置。
    对 (j1,j2) 对最多 2**8 = 256 个局部配置，完全精确，无任何近似。

    B_sub_cur : (K_cur, K_mid)  mid -> cur 权重
    B_sub_mid : (K_mid, K_ref)  ref -> mid 权重
    propA_ref : (2**K_ref,)     ref 层完整 pmf
    """
    K_cur, K_mid = B_sub_cur.shape
    deps_cur = [np.where(B_sub_cur[j] != 0)[0] for j in range(K_cur)]
    deps_mid = [np.where(B_sub_mid[k] != 0)[0] for k in range(K_mid)]

    mu    = np.zeros(K_cur)
    Sigma = np.zeros((K_cur, K_cur))

    # 一阶矩
    for j in range(K_cur):
        dep = list(deps_cur[j])
        if not dep:
            mu[j] = _logistic(intercept_cur[j])
            continue
        jt    = _compute_local_joint(dep, propA_ref, A_ref, B_sub_mid, intercept_mid)
        ac    = _binary_matrix(len(dep))
        w     = B_sub_cur[j][dep]
        mu[j] = (_logistic(intercept_cur[j] + ac @ w) * jt).sum()

    np.fill_diagonal(Sigma, mu)   # Bernoulli：E[L_k²] = E[L_k]

    # 二阶矩
    for j1 in range(K_cur):
        for j2 in range(j1 + 1, K_cur):
            d1    = set(deps_cur[j1]);  d2 = set(deps_cur[j2])
            union = sorted(d1 | d2)
            # 通过 mid->ref 检测两变量是否完全独立（无任何共同祖先）
            k2_d1 = set(c for k in d1 for c in deps_mid[k])
            k2_d2 = set(c for k in d2 for c in deps_mid[k])
            if not k2_d1 & k2_d2 and not d1 & d2:
                Sigma[j1, j2] = Sigma[j2, j1] = mu[j1] * mu[j2]
                continue
            jt  = _compute_local_joint(union, propA_ref, A_ref, B_sub_mid, intercept_mid)
            ac  = _binary_matrix(len(union))
            i1  = [union.index(k) for k in sorted(d1)]
            i2  = [union.index(k) for k in sorted(d2)]
            w1  = B_sub_cur[j1][sorted(d1)];  w2 = B_sub_cur[j2][sorted(d2)]
            l1  = intercept_cur[j1] + ac[:, i1] @ w1
            l2  = intercept_cur[j2] + ac[:, i2] @ w2
            Sigma[j1, j2] = Sigma[j2, j1] = (_logistic(l1) * _logistic(l2) * jt).sum()

    return mu, Sigma


# ─── 主函数 ──────────────────────────────────────────────────────────────────

def build_true_params(K_list, J, max_val):
    """
    优化版 build_true_params，支持 D=1 至 D=5。

    核心策略（三档自动选择）
    ─────────────────────────
    令 pivot = 最浅（下标最小）的可枚举层（K_list[pivot] ≤ 22，即 2**K ≤ 4M）。
    从顶层向下传播完整 pmf 到 pivot 层，再按 pivot 值选路径：

    ┌─────────┬──────────────────────────────────────────────────────────────┐
    │ pivot=0 │ K_list[0]≤22（D=1,2,3）                                     │
    │         │ propA 已到 K_list[0]，直接用精确 pmf 算矩，O(2**K_list[0]) │
    ├─────────┼──────────────────────────────────────────────────────────────┤
    │ pivot=1 │ K_list[0]>22，K_list[1]≤22（D=4: K=[54,18,6,2]）           │
    │         │ Dense EXP_1 (K0 × 2**K1)，如 D=4 时约 113 MB               │
    ├─────────┼──────────────────────────────────────────────────────────────┤
    │ pivot=2 │ K_list[1]>22，K_list[2]≤22（D=5: K=[162,54,18,6,2]）       │
    │         │ _compute_moments_sparse 跳过 K_list[1]：                    │
    │         │   利用 B_sub 稀疏性（每行≤2非零），局部枚举 ≤2**8=256 点，  │
    │         │   精确计算 K_list[0] 的一/二阶矩，峰值内存 O(2**K_list[2]) │
    └─────────┴──────────────────────────────────────────────────────────────┘

    Parameters（与原版相同）
    ─────────────────────────
    K_list  : list[int]   [K1, K2, ..., KL]，通常满足 K_l = 3·K_{l+1}
    J       : int         观测变量维度
    max_val : float       参数最大值

    Returns（与原版相同）
    ─────────────────────
    prop_true, B_list, G_list, gamma
    """
    K_list = list(K_list)
    L = len(K_list)
    prop_true = 0.5 * np.ones(K_list[-1])

    # ── 构造 B_sub 和 higher_B_list（与原版完全相同）────────────────────────
    def _bsub(nr, nc):
        B = np.zeros((nr, nc)); half = int(np.floor(nc / 2))
        for k in range(nc):
            if (k + 1) + half <= nc:
                r1, r2 = (k+1)+2*nc-1, (k+1)+2*nc+half-1
                c1, c2 = k, k + half
                if r1 < nr and r2 < nr and c2 < nc:
                    B[r1, c1] =  max_val;  B[r2, c1] = -max_val / 3
                    B[r1, c2] = -max_val / 3;  B[r2, c2] =  max_val
            if k + nc < nr:
                B[[k, k + nc], k] = max_val
        return B

    B_sub_list = [_bsub(J, K_list[0])]
    higher_B_list = []; B_sub_higher = []
    for l in range(1, L):
        nr, nc = K_list[l - 1], K_list[l]; bs = _bsub(nr, nc)
        ic = np.concatenate([
            -max_val / 2 * np.ones(nc),
            -max_val     * np.ones(nc),
            -max_val / 2 * np.ones(nc),
        ])
        higher_B_list.append(np.column_stack([ic, bs]))
        B_sub_higher.append(bs)
    ic_higher = [B[:, 0] for B in higher_B_list]   # 各层截距

    # ── 步骤1：从顶层向下传播 pmf 到 pivot 层 ────────────────────────────────
    MAX_ENUM = 22   # 可枚举的最大层宽（2**22 ≈ 4M，约 32 MB）
    pivot = next(i for i in range(L) if K_list[i] <= MAX_ENUM)

    K_U = K_list[-1]
    A_U, propA_U = _compute_propA_top(prop_true)
    for i in range(L - 2, pivot - 1, -1):
        K_L   = K_list[i]
        EXP_L = _logistic(higher_B_list[i] @ np.vstack([np.ones(2**K_U), A_U.T]))
        propA_U = _marginal_next_layer(EXP_L, propA_U)
        K_U   = K_L;  A_U = _binary_matrix(K_L)
    # propA_U 在 K_list[pivot] 层

    # ── 步骤2：计算 K_list[0] 层的一/二阶矩 ─────────────────────────────────
    if pivot == 0:
        # K_list[0] 本身可枚举：直接从 propA_U 计算精确矩
        mu1    = A_U.T @ propA_U
        Sigma2 = (A_U.T * propA_U[None, :]) @ A_U
        np.fill_diagonal(Sigma2, mu1)

    elif pivot == 1:
        # Dense：EXP_1 shape (K_list[0], 2**K_list[1])
        EXP_1  = _logistic(higher_B_list[0] @ np.vstack([np.ones(2**K_list[1]), A_U.T]))
        mu1    = EXP_1 @ propA_U
        Sigma2 = (EXP_1 * propA_U[None, :]) @ EXP_1.T
        np.fill_diagonal(Sigma2, mu1)

    else:
        # pivot == 2：稀疏两层跳（ref=K_list[2], mid=K_list[1], cur=K_list[0]）
        assert pivot == 2, (
            f"pivot={pivot} > 2 not yet implemented. "
            f"K_list[1]={K_list[1]} > 2^{MAX_ENUM}. "
            "Consider extending _compute_moments_sparse to multi-hop."
        )
        mu1, Sigma2 = _compute_moments_sparse(
            B_sub_higher[0], ic_higher[0],   # cur 层的连接（K_list[0] × K_list[1]）
            B_sub_higher[1], ic_higher[1],   # mid 层的连接（K_list[1] × K_list[2]）
            propA_U, A_U                     # ref 层（K_list[2]）的完整 pmf
        )

    # ── 步骤3：构造 B1 ────────────────────────────────────────────────────────
    W  = B_sub_list[0]       # (J, K1)
    b0 = -W @ mu1            # (J,)
    B1 = np.column_stack([b0, W])

    # ── 步骤4：E[μⱼ²] = -b0ⱼ² + (W Σ Wᵀ)[j,j] ───────────────────────────────
    diag_WSW        = np.einsum('jk,kl,jl->j', W, Sigma2, W)
    one_minus_sigma2 = -b0**2 + diag_WSW

    # ── 步骤5：缩放（与原版相同）──────────────────────────────────────────────
    idx = one_minus_sigma2 > 0.75
    if idx.any():
        scale = np.sqrt(0.75 / one_minus_sigma2[idx, None])
        B1[idx, :] *= scale
        b0_s, W_s = B1[:, 0], B1[:, 1:]
        one_minus_sigma2 = -b0_s**2 + np.einsum('jk,kl,jl->j', W_s, Sigma2, W_s)

    gamma  = 1.0 - one_minus_sigma2
    B_list = [B1] + higher_B_list
    G_list = [(b != 0) for b in [B_sub_list[0]] + B_sub_higher]
    return prop_true, B_list, G_list, gamma

# ─── IdCondCheck ─────────────────────────────────────────────────────────────

def _compute_propA_U(prop: np.ndarray, A_U: np.ndarray) -> np.ndarray:
    """顶层独立 Bernoulli 联合 pmf。"""
    p   = np.asarray(prop, dtype=float)
    lp  = np.log(np.clip(p,       1e-12, 1 - 1e-12))
    l1p = np.log(np.clip(1.0 - p, 1e-12, 1 - 1e-12))
    r   = np.exp(A_U @ lp + (1.0 - A_U) @ l1p)
    return r / r.sum()


def IdCondCheck_fast(D, prop, K_list, B_list, Var, batch_size=512, MAX_ENUM=22):
    """
    优化版 IdCondCheck — 支持任意大 K_list（包括 D=5）。

    原版在 K_list 较大时（如 D=5，K_list[1]=162）需要传播
    2**162 维的完整 pmf，直接导致 MemoryError。

    优化策略
    ────────
    令 pivot = 最浅可枚举层（最小的 d∈[1,D] 使 K_list[d] ≤ MAX_ENUM）。
    将传播链分两段：

    段A（完整 pmf 传播，精确）
        K_list[D] → … → K_list[pivot]
        每步用 _marginal_next_layer 批量计算，内存 O(batch_size × 2**K_list[d+1])。

    段B（矩方法，精确，不存完整 pmf）
        从 propA 在 K_list[pivot] 层出发，计算 K_list[1] 的一/二阶矩 (μ₁, Σ₁)：

        ┌─────────┬────────────────────────────────────────────────────────┐
        │ pivot=1 │ propA 已在 K_list[1]，直接从精确 pmf 算矩             │
        ├─────────┼────────────────────────────────────────────────────────┤
        │ pivot=2 │ Dense EXP 一步跳：(K_list[1] × 2**K_list[2])          │
        │         │ 如 D=4 时约 113 MB，精确                               │
        ├─────────┼────────────────────────────────────────────────────────┤
        │ pivot=3 │ 稀疏两步跳（_compute_moments_sparse）：                │
        │         │ ref=K_list[3], mid=K_list[2], cur=K_list[1]           │
        │         │ 利用 B_list[1,2] 每行≤2非零列，局部枚举≤2**8=256点    │
        │         │ 精确，内存峰值 O(2**K_list[3])                        │
        └─────────┴────────────────────────────────────────────────────────┘

    最终输出
    ────────
    B_list[0] is None 且 K_list[1] 可枚举（pivot=1）
        → propA_U (2**K_list[1],)                    [原始行为，完全兼容]
    B_list[0] is None 且 K_list[1] 不可枚举（pivot≥2）
        → {"mu1": (K1,), "Sigma2": (K1,K1)}
    B_list[0] 不为 None
        → {"weighted1M": (J,), "weighted2M": (J,)}   [精确，用矩方法]
          weighted1M[j] = b0[j] + W[j,:]·μ₁
          weighted2M[j] = Var[j] + b0[j]² + 2·b0[j]·(W[j,:]·μ₁) + (WΣ₁Wᵀ)[j,j]

    Parameters（与原版相同，新增可选参数）
    ────────────────────────────────────
    D          : int            潜在层数
    prop       : array-like     顶层 Bernoulli 参数，长度 K_list[D]
    K_list     : list           各层维度；K_list[0]=J（观测层），K_list[D]=顶层
    B_list     : list[ndarray]  层间矩阵；B_list[d] shape (K_list[d], K_list[d+1]+1)
    Var        : array-like     观测层方差，长度 J（B_list[0] is None 时可为 None）
    batch_size : int            _marginal_next_layer 的批大小（默认 512）
    MAX_ENUM   : int            可枚举层宽阈值（默认 22，即 2**22 ≈ 4M）
    """
    # ── 段A：从顶层传播 pmf 到 pivot 层 ──────────────────────────────────────
    pivot   = next(d for d in range(1, D + 1) if K_list[d] <= MAX_ENUM)
    K_U     = K_list[D]
    A_U     = _binary_matrix(K_U)
    propA_U = _compute_propA_U(prop, A_U)

    for d in range(D - 1, pivot - 1, -1):
        K_L     = K_list[d]
        EXP_L   = _logistic(B_list[d] @ np.vstack([np.ones(2**K_U), A_U.T]))
        propA_U = _marginal_next_layer(EXP_L, propA_U, batch_size=batch_size)
        K_U     = K_L
        A_U     = _binary_matrix(K_L)
    # propA_U 在 K_list[pivot] 层

    # ── 段B：计算 K_list[1] 的一/二阶矩 ──────────────────────────────────────
    propA_1_exact = None   # 仅 pivot==1 时有效

    if pivot == 1:
        # propA 已在 K_list[1]，直接算精确矩
        mu1    = A_U.T @ propA_U                             # (K1,)
        Sigma2 = (A_U.T * propA_U[None, :]) @ A_U           # (K1, K1)
        np.fill_diagonal(Sigma2, mu1)
        propA_1_exact = propA_U

    elif pivot == 2:
        # Dense 一步：K_list[2]（可枚举）→ K_list[1]
        # B_list[1] shape (K_list[1], K_list[2]+1)
        EXP_1  = _logistic(B_list[1] @ np.vstack([np.ones(2**K_U), A_U.T]))
        mu1    = EXP_1 @ propA_U                             # (K1,)
        Sigma2 = (EXP_1 * propA_U[None, :]) @ EXP_1.T       # (K1, K1)
        np.fill_diagonal(Sigma2, mu1)

    else:
        # pivot == 3：稀疏两步跳
        # ref = K_list[3]，mid = K_list[2]，cur = K_list[1]
        # 要求 B_list[1] 和 B_list[2] 每行 ≤ 2 个非零权重列
        assert pivot == 3, (
            f"pivot={pivot} > 3 not yet implemented. "
            f"K_list[2]={K_list[2]} > 2^{MAX_ENUM}; "
            "consider increasing MAX_ENUM or extending _compute_moments_sparse."
        )
        mu1, Sigma2 = _compute_moments_sparse(
            B_list[pivot - 2][:, 1:], B_list[pivot - 2][:, 0],  # cur 层
            B_list[pivot - 1][:, 1:], B_list[pivot - 1][:, 0],  # mid 层
            propA_U, A_U                                          # ref 层 pmf
        )

    # ── 输出 ──────────────────────────────────────────────────────────────────
    if B_list[0] is None:
        # 原始接口：返回 propA（若可枚举）或矩（若不可枚举）
        if propA_1_exact is not None:
            return propA_1_exact
        return {"mu1": mu1, "Sigma2": Sigma2}

    # 用矩方法精确计算 weighted1M/2M（避免构造 (J, 2**K1) 的 mu 矩阵）
    b0 = B_list[0][:, 0]    # (J,)
    W  = B_list[0][:, 1:]   # (J, K1)
    weighted1M = b0 + W @ mu1
    weighted2M = (np.asarray(Var)
                  + b0**2
                  + 2.0 * b0 * (W @ mu1)
                  + np.einsum('jk,kl,jl->j', W, Sigma2, W))
    return {"weighted1M": weighted1M, "weighted2M": weighted2M}