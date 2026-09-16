# generate_X_bernoulli.py; generate_X_normal.py; generate_X_poisson.py

# utilities/generate_X_bernoulli.py
import numpy as np
from scipy.special import expit  # logistic sigmoid

from .generate_copula_D import simulate_latent_layers_D

def generate_X_bernoulli(N, prop_true, B1, B2):
    """
    Generate binary data according to a hierarchical Bernoulli model.
    Equivalent to Matlab generate_X_bernoulli.m

    Parameters
    ----------
    N : int
        Number of samples.
    prop_true : array_like
        Probabilities for generating A2 (length K2).
    B1 : np.ndarray
        Coefficient matrix for generating X (shape: K1 x (1+K1_from_A1)).
    B2 : np.ndarray
        Coefficient matrix for generating A1 (shape: K1_from_A1 x (1+K2)).
    
    Returns
    -------
    X : np.ndarray
        Generated binary data (N x number_of_columns_in_B1).
    A1 : np.ndarray
        Intermediate latent binary variables (N x B2.shape[0]).
    A2 : np.ndarray
        Bottom latent binary variables (N x len(prop_true)).
    """

    prop_true = np.asarray(prop_true)
    K2 = prop_true.shape[0]

    # --- Generate A2 ---
    A2 = np.zeros((N, K2))
    for k in range(K2):
        A2[:, k] = np.random.binomial(1, prop_true[k], size=N)

    # --- Generate A1 ---
    logits_A1 = np.dot(np.column_stack((np.ones(N), A2)), B2.T)
    probs_A1 = expit(logits_A1)
    A1 = np.random.binomial(1, probs_A1)

    # --- Generate X ---
    logits_X = np.dot(np.column_stack((np.ones(N), A1)), B1.T)
    probs_X = expit(logits_X)
    X = np.random.binomial(1, probs_X)

    return X, A1, A2

# utilities/generate_X_normal.py

# def generate_X_normal(N, prop_true, B1, B2, gamma):
#     """
#     Generate data where latent binary variables influence Gaussian observations.
#     Equivalent to Matlab generate_X_normal.m.

#     Parameters
#     ----------
#     N : int
#         Number of samples.
#     prop_true : array_like
#         Probabilities for generating A2 (length K2).
#     B1 : np.ndarray
#         Coefficient matrix for X generation (shape: p x (1+K1)).
#     B2 : np.ndarray
#         Coefficient matrix for A1 generation (shape: K1 x (1+K2)).
#     gamma : array_like
#         Variances for each Gaussian variable (length p).

#     Returns
#     -------
#     X : np.ndarray
#         Gaussian observations (N x p).
#     A1 : np.ndarray
#         Latent binary matrix (N x K1).
#     A2 : np.ndarray
#         Latent binary matrix (N x K2).
#     """

#     prop_true = np.asarray(prop_true)
#     K2 = prop_true.shape[0]

#     # --- Generate A2 ---
#     A2 = np.zeros((N, K2))
#     for k in range(K2):
#         A2[:, k] = np.random.binomial(1, prop_true[k], size=N)

#     # --- Generate A1 ---
#     logits_A1 = np.dot(np.column_stack((np.ones(N), A2)), B2.T)
#     probs_A1 = expit(logits_A1)
#     ## MARK: chop
#     # probs_A1 = np.clip(expit(logits_A1), 1e-8, 1 - 1e-8)
#     A1 = np.random.binomial(1, probs_A1)

#     # --- Generate X ---
#     mean_X = np.dot(np.column_stack((np.ones(N), A1)), B1.T)
#     std_X = np.sqrt(np.asarray(gamma)).reshape(1, -1)  # ensure shape (1, p)
#     X = np.random.normal(loc=mean_X, scale=std_X, size=mean_X.shape)

#     return X, A1, A2

def generate_X_normal(N, prop_true, B_list, gamma):
    """
    Generate data for Normal-D-latent-layer DDEs.

    Parameters
    ----------
    N        : int
    prop_true : (K_D,)  Bernoulli probs for top layer
    B_list   : list of D arrays
                 B_list[0] : (J,  K_1+1)   X  <- A_1
                 B_list[d] : (K_d, K_{d+1}+1)  A_d <- A_{d+1},  d=1,...,D-1
    gamma    : (J,)  conditional variances for X

    Returns
    -------
    X          : (N, J)
    A_list     : list of D arrays, A_list[d] is (N, K_{d+1}),
                 ordered [A_1, A_2, ..., A_D]
    """
    prop_true = np.asarray(prop_true)
    D = len(B_list)
    K_D = prop_true.shape[0]
    ones = np.ones((N, 1))

    A_list = [None] * D   # will store A_1, ..., A_D

    # --- Generate top layer A_D ~ Bernoulli(prop_true) ---
    A_D = np.zeros((N, K_D))
    for k in range(K_D):
        A_D[:, k] = np.random.binomial(1, prop_true[k], size=N)
    A_list[D - 1] = A_D

    # --- Generate A_{D-1}, ..., A_1 top-down ---
    # B_list[d] : (K_d, K_{d+1}+1)  generates A_d given A_{d+1}
    for d in range(D - 2, -1, -1):   # d = D-2, D-3, ..., 0
        A_parent = A_list[d + 1]                              # (N, K_{d+2})  -- already generated
        logits = np.hstack([ones, A_parent]) @ B_list[d + 1].T  # (N, K_{d+1})
        probs  = expit(logits)
        A_list[d] = np.random.binomial(1, probs).astype(float)

    # --- Generate X ~ Normal(B_list[0] @ [1, A_1], gamma) ---
    mean_X = np.hstack([ones, A_list[0]]) @ B_list[0].T      # (N, J)
    std_X  = np.sqrt(np.asarray(gamma)).reshape(1, -1)
    X = np.random.normal(loc=mean_X, scale=std_X)

    return X, A_list

# utilities/generate_X_poisson.py

def generate_X_poisson(N, prop_true, B_list, random_state=None):
    """Generate observations from an arbitrary-depth Poisson DDE.

    ``B_list`` follows the package convention: the first matrix maps ``A_1``
    to the Poisson log-rate, and each remaining matrix maps a binary parent
    layer to the binary layer immediately below it.

    Parameters
    ----------
    N : int
        Number of samples.
    prop_true : array_like
        Deepest-layer Bernoulli probabilities.
    B_list : sequence of np.ndarray
        Arbitrary-depth DDE coefficient matrices.
    random_state : optional
        Seed or NumPy generator.

    Returns
    -------
    X : np.ndarray
        Poisson-distributed data (N x p).
    A_list : list of np.ndarray
        Latent binary matrices ordered from shallowest to deepest.
    """

    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)
    B_list = [np.asarray(B, dtype=float) for B in B_list]
    if not B_list:
        raise ValueError("B_list must contain at least one latent layer.")
    N = int(N)
    if N < 1:
        raise ValueError("N must be a positive integer.")

    A_list, _ = simulate_latent_layers_D(N, prop_true, B_list, rng)
    design = np.column_stack((np.ones(N), A_list[0]))
    eta = design @ B_list[0].T
    with np.errstate(over="ignore"):
        rate = np.exp(eta)
    if not np.all(np.isfinite(rate)) or np.any(rate > 1.0e18):
        raise ValueError(
            "Poisson natural parameters produce rates too large to sample."
        )
    X = rng.poisson(rate, size=rate.shape)
    return X, A_list

# For 1 layer DDE model

def generate_X_normal_single_layer(N, prop_true, B1, gamma):
    """
    Generate data with one layer of latent binary variables influencing Gaussian observations.

    Parameters
    ----------
    N : int
        Number of samples.
    prop_true : array_like
        Probabilities for generating latent variables A (length K).
    B1 : np.ndarray
        Coefficient matrix for X generation (shape: p x (1+K)).
        The first column is intercepts.
    gamma : array_like
        Variances for each Gaussian variable (length p).

    Returns
    -------
    X : np.ndarray
        Gaussian observations (N x p).
    A1 : np.ndarray
        Latent binary matrix (N x K).
    """
    prop_true = np.asarray(prop_true)
    K1 = prop_true.shape[0]

    # --- Generate latent A1 ---
    A1 = np.zeros((N, K1))
    for k in range(K1):
        A1[:, k] = np.random.binomial(1, prop_true[k], size=N)

    # --- Generate X ---
    mean_X = np.dot(np.column_stack((np.ones(N), A1)), B1.T)
    std_X = np.sqrt(np.asarray(gamma)).reshape(1, -1)
    X = np.random.normal(loc=mean_X, scale=std_X, size=mean_X.shape)

    return X, A1
