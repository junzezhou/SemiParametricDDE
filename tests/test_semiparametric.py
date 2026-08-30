import numpy as np
from scipy.stats import norm

from dde.utils.semiparametric import g_est_rversion


def legacy_g_est_rversion(y, z, n, p, K, pro0, delta0):
    y = np.asarray(y, dtype=float)
    z_idx = np.asarray(z, dtype=int).reshape(-1)
    pro0 = np.asarray(pro0, dtype=float).reshape(-1)

    one_minus_delta0 = 1.0 - delta0
    counts = np.bincount(z_idx[z_idx >= 0], minlength=K).astype(float)
    inv_counts = np.zeros_like(counts)
    nonzero = counts > 0
    inv_counts[nonzero] = 1.0 / counts[nonzero]

    prob1 = np.zeros((K, n, p), dtype=float)
    y_rank = np.zeros((n, p), dtype=int)
    z_sort = np.zeros((n, p), dtype=int)

    for j in range(p):
        y_rank[:, j] = np.argsort(np.argsort(y[:, j], kind="mergesort"), kind="mergesort")
        order_j = np.argsort(y[:, j], kind="mergesort")
        z_sort[:, j] = z_idx[order_j]

    ct = np.zeros((n, K), dtype=int)
    for j in range(p):
        ct[0, :] = 0
        if z_sort[0, j] >= 0:
            ct[0, z_sort[0, j]] = 1
        for i in range(1, n):
            ct[i, :] = ct[i - 1, :]
            if z_sort[i, j] >= 0:
                ct[i, z_sort[i, j]] += 1
        for i in range(n):
            prob1[:, i, j] = ct[y_rank[i, j], :] * inv_counts

    prob1 = np.clip(prob1, delta0, one_minus_delta0)
    qnorm = norm.ppf(prob1)
    g_est = np.zeros((n, p))

    for j in range(p):
        g_col = pro0 @ qnorm[:, :, j]
        g_col = (g_col - np.mean(g_col)) / (np.std(g_col) + 1e-4)
        g_est[:, j] = g_col

    return g_est


def test_g_est_rversion_matches_legacy_probability_cube():
    rng = np.random.default_rng(123)

    for case in range(20):
        n = int(rng.integers(8, 40))
        p = int(rng.integers(1, 8))
        K = int(rng.integers(4, 20))
        y = rng.normal(size=(n, p))
        if case % 3 == 0:
            y = np.round(y, 1)

        z = rng.integers(-1, K, size=n)
        if case % 4 == 0:
            z[z == 0] = min(K - 1, 3)

        pro0 = rng.random(K)
        pro0 = pro0 / pro0.sum()

        expected = legacy_g_est_rversion(y, z, n, p, K, pro0, 1e-6)
        actual = g_est_rversion(y, z, n, p, K, pro0, 1e-6)

        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_g_est_rversion_handles_large_unobserved_class_space():
    rng = np.random.default_rng(456)
    n = 300
    p = 4
    K = 100_000
    observed = rng.choice(K, size=200, replace=False)
    z = rng.choice(observed, size=n)
    y = rng.normal(size=(n, p))
    pro0 = rng.random(K)
    pro0 = pro0 / pro0.sum()

    actual = g_est_rversion(y, z, n, p, K, pro0, 1e-6)

    assert actual.shape == (n, p)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual.mean(axis=0), np.zeros(p), atol=1e-12)
