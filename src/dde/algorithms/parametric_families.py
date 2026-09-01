"""Observed-family adapters for parametric DDE estimators.

The latent hierarchy is common to every parametric DDE family.  Adapters in
this module own only observed-layer behavior: data validation and
initialization, the observed contribution to latent conditional log-odds,
observed-layer M-steps, dispersion semantics, scoring, and simulation.

Only the Normal adapter is registered during the behavior-preserving
refactor.  Bernoulli, Poisson, and Lognormal adapters can therefore be added
without cloning either the fixed-width SAEM or CSP-MCEM control loops.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize

from dde.utils import TLP, thres

from .normal_init_semi import Normal_init_semi


def _tlp_penalty(x, pen, tau):
    """Return the paper/MATLAB TLP penalty ``pen * sum(min(abs(x), tau))``."""
    return float(pen) * TLP(x, tau)


def _tlp_penalty_value_grad(x, pen, tau):
    """Return the TLP value and a valid gradient away from its knots."""
    x = np.asarray(x, dtype=float)
    value = _tlp_penalty(x, pen, tau)
    grad = np.zeros_like(x)
    active = (np.abs(x) < tau) & (x != 0.0)
    grad[active] = float(pen) * np.sign(x[active])
    return value, grad


class _ObservedNormalAccumulator:
    """Layer-shared running quadratic statistics for Normal responses."""

    def __init__(self, n_responses, n_features):
        self.Q = np.zeros((n_features, n_features), dtype=float)
        self.q_all = np.zeros((n_responses, n_features), dtype=float)

    def update(self, X, A_samples, t):
        alpha = 1.0 / t
        Q_new, q_new = _normal_sample_statistics(X, A_samples)
        self.Q = (1.0 - alpha) * self.Q + alpha * Q_new
        self.q_all = (1.0 - alpha) * self.q_all + alpha * q_new
        return Q_new, q_new

    def value(self, response_index, x):
        q = self.q_all[response_index]
        return float(x @ self.Q @ x - 2.0 * q @ x)

    def value_and_grad(self, response_index, x):
        q = self.q_all[response_index]
        value = float(x @ self.Q @ x - 2.0 * q @ x)
        grad = 2.0 * (self.Q @ x - q)
        return value, grad


def _normal_sample_statistics(X, A_samples):
    """Return current-MC Normal design cross-products, averaged over draws."""
    X = np.asarray(X, dtype=float)
    A_samples = np.asarray(A_samples, dtype=float)
    C = A_samples.shape[2]
    Q = np.einsum("nkc,nlc->kl", A_samples, A_samples, optimize=True) / C
    q_all = X.T @ np.mean(A_samples, axis=2)
    return Q, q_all


def _update_deep_normal_observed(X, Q, q_all, tau, *, x_sq_sum=None):
    """MATLAB D>=3 closed-form observed-layer update and variances."""
    X = np.asarray(X, dtype=float)
    try:
        coefficients_t = np.linalg.solve(Q, q_all.T)
    except np.linalg.LinAlgError:
        coefficients_t = np.linalg.lstsq(Q, q_all.T, rcond=1e-10)[0]
    B_update = coefficients_t.T
    B_update[:, 1:] = thres(B_update[:, 1:], tau)

    quadratic = np.einsum("jp,pq,jq->j", B_update, Q, B_update, optimize=True)
    cross = np.einsum("jp,jp->j", B_update, q_all, optimize=True)
    if x_sq_sum is None:
        x_sq_sum = np.sum(X**2, axis=0)
    residual_sum_squares = x_sq_sum - 2.0 * cross + quadratic
    gamma_update = residual_sum_squares / X.shape[0]
    return B_update, gamma_update


def _soft_threshold(x, threshold):
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def _largest_eigenvalue_psd(matrix: np.ndarray) -> float:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return 0.0
    return max(float(np.linalg.eigvalsh(matrix)[-1]), 0.0)


def _solve_gaussian_weighted_l1_from_statistics(
    X,
    Q,
    q_all,
    B_start,
    gamma,
    penalty_scale,
    temperature,
    *,
    mc_samples=1,
    x_sq_sum=None,
    max_iter=100,
    tol=1e-6,
    bound=10.0,
):
    """Batched Normal-layer update from MC or SA sufficient statistics."""
    X = np.asarray(X, dtype=float)
    Q = np.asarray(Q, dtype=float)
    q_all = np.asarray(q_all, dtype=float)
    B = np.asarray(B_start, dtype=float).copy()
    gamma = np.maximum(np.asarray(gamma, dtype=float).reshape(-1), 1e-10)
    penalty_scale = np.asarray(penalty_scale, dtype=float)
    C = int(mc_samples)

    largest = _largest_eigenvalue_psd(Q)
    if largest <= 1e-14:
        B[:, 0] = np.clip(np.mean(X, axis=0), -bound, bound)
        B[:, 1:] = 0.0
    else:
        row_step = gamma / (float(temperature) * largest)
        row_step = np.maximum(row_step, 1e-14)
        Y = B.copy()
        momentum = 1.0
        for _ in range(int(max_iter)):
            gradient = (float(temperature) / gamma[:, None]) * (Y @ Q - q_all)
            candidate = Y - row_step[:, None] * gradient
            candidate[:, 0] = np.clip(candidate[:, 0], -bound, bound)
            candidate[:, 1:] = _soft_threshold(
                candidate[:, 1:],
                row_step[:, None] / penalty_scale,
            )
            candidate[:, 1:] = np.clip(candidate[:, 1:], -bound, bound)

            relative = np.linalg.norm(candidate - B) / (np.linalg.norm(B) + 1e-12)
            next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum**2))
            Y = candidate + ((momentum - 1.0) / next_momentum) * (candidate - B)
            B = candidate
            momentum = next_momentum
            if relative <= tol:
                break

    quadratic = np.einsum("jp,pq,jq->j", B, Q, B, optimize=True)
    cross = np.einsum("jp,jp->j", B, q_all, optimize=True)
    if x_sq_sum is None:
        x_sq_sum = np.sum(X**2, axis=0)
    else:
        x_sq_sum = np.asarray(x_sq_sum, dtype=float).reshape(-1)
        if x_sq_sum.shape != (X.shape[1],):
            raise ValueError("x_sq_sum must have one entry per observed column.")
    sse_average = np.maximum(x_sq_sum - 2.0 * cross + quadratic, 0.0)
    sse = C * sse_average
    N = X.shape[0]
    gamma_update = (1.0 + 0.5 * sse) / (1.0 + 0.5 * N * C + 1.0)
    return B, np.maximum(gamma_update, 1e-10)


def _solve_gaussian_weighted_l1(
    X,
    A_sample,
    B_start,
    gamma,
    penalty_scale,
    temperature,
    *,
    x_sq_sum=None,
    max_iter=100,
    tol=1e-6,
    bound=10.0,
):
    """Batched FISTA update for one current Normal Monte Carlo sample."""
    Q, q_all = _normal_sample_statistics(X, A_sample)
    return _solve_gaussian_weighted_l1_from_statistics(
        X,
        Q,
        q_all,
        B_start,
        gamma,
        penalty_scale,
        temperature,
        mc_samples=A_sample.shape[2],
        x_sq_sum=x_sq_sum,
        max_iter=max_iter,
        tol=tol,
        bound=bound,
    )


@dataclass(frozen=True)
class NormalFamilyAdapter:
    """Observed-layer operations for a conditionally Normal DDE."""

    name: str = "Normal"
    key: str = "normal"
    aliases: tuple[str, ...] = ("normal", "gaussian")
    has_dispersion: bool = True
    dispersion_name: str = "conditional_variance"

    def initial_working_response(self, Y, *, epsilon) -> np.ndarray:
        """Return the response used by the shared spectral initializer.

        Normal uses the identity mean link.  Discrete-family adapters can
        instead return a stabilized inverse-mean-link response while keeping
        the latent spectral initialization outside the fitting runners.
        """
        del epsilon
        return self.validate_y(Y).copy()

    def validate_y(self, Y) -> np.ndarray:
        X = np.asarray(Y, dtype=float)
        if X.ndim != 2:
            raise ValueError(
                f"Y must be a two-dimensional matrix; got shape {X.shape}."
            )
        if X.shape[0] < 1 or X.shape[1] < 1:
            raise ValueError("Y must contain at least one observation and one variable.")
        if not np.all(np.isfinite(X)):
            raise ValueError("Normal DDE requires finite values in Y.")
        return X

    def validate_dispersion(self, dispersion, n_responses: int) -> np.ndarray:
        values = np.asarray(dispersion, dtype=float).reshape(-1)
        if values.shape != (int(n_responses),):
            raise ValueError(
                f"gamma shape {values.shape} must be ({int(n_responses)},)."
            )
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("gamma entries must be finite and positive.")
        return values

    def initialize(
        self,
        X,
        K_list,
        *,
        epsilon,
        mode,
        B_list_true=None,
        G_list_true=None,
    ):
        working_response = self.initial_working_response(X, epsilon=epsilon)
        if mode == "data":
            return Normal_init_semi(
                working_response,
                K_list[0],
                K_list,
                epsilon=epsilon,
                mode="data",
            )
        return Normal_init_semi(
            working_response,
            K_list[0],
            K_list,
            B_list_true,
            G_list_true,
            epsilon,
            mode="oracle",
        )

    def observed_conditional_logit(
        self,
        X,
        B1,
        dispersion,
        A1,
        *,
        width,
        row_chunk_size,
        coordinate_chunk_size=1,
    ) -> np.ndarray:
        """Return Normal observed-layer log-likelihood odds for ``A_1``."""
        del coordinate_chunk_size
        X = np.asarray(X, dtype=float)
        gamma = np.asarray(dispersion, dtype=float)
        B1 = np.asarray(B1, dtype=float)
        A1 = np.asarray(A1)
        N = X.shape[0]
        slopes = B1[:, 1 : width + 1]
        weighted_slopes = slopes / gamma[:, None]
        slope_gram = slopes.T @ weighted_slopes
        slope_norms = np.diag(slope_gram)
        like_logit = np.empty((N, width), dtype=float)
        for start in range(0, N, row_chunk_size):
            stop = min(start + row_chunk_size, N)
            A_chunk = A1[start:stop, :width]
            data_linear = (
                (X[start:stop] - B1[:, 0][None, :]) / gamma[None, :]
            ) @ slopes
            like_logit[start:stop] = (
                data_linear
                - A_chunk @ slope_gram
                + (A_chunk - 0.5) * slope_norms[None, :]
            )
        return like_logit

    def make_fixed_accumulator(self, n_responses, n_features):
        return _ObservedNormalAccumulator(n_responses, n_features)

    def make_response_cache(self, X):
        """Cache response-only quantities shared by observed M-steps."""
        return np.sum(np.asarray(X, dtype=float) ** 2, axis=0)

    def copy_dispersion(self, dispersion):
        return np.asarray(dispersion, dtype=float).copy()

    def fixed_mstep(
        self,
        X,
        A_samples,
        B_start,
        dispersion,
        accumulator,
        *,
        iteration,
        depth,
        penalty,
        tau,
        bound_val,
        response_cache,
    ):
        """Perform the existing fixed-width Normal observed-layer update."""
        N, J = X.shape
        K1 = B_start.shape[1] - 1
        C = A_samples.shape[2]
        Q_current, q_current = accumulator.update(X, A_samples, iteration)
        if depth >= 3:
            return _update_deep_normal_observed(
                X,
                Q_current,
                q_current,
                tau,
                x_sq_sum=response_cache,
            )

        B_update = np.zeros_like(B_start)
        gamma_update = np.zeros(J)
        for j in range(J):
            Xj = X[:, j]

            def obj(x, __j=j, __pen=penalty):
                value, grad = accumulator.value_and_grad(__j, x)
                pen_value, pen_grad = _tlp_penalty_value_grad(
                    x[1:], __pen, tau
                )
                grad[1:] += pen_grad
                return value + pen_value, grad

            bounds = [(-bound_val, bound_val)] * (K1 + 1)
            res = minimize(
                obj,
                B_start[j, :],
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={"maxiter": 10},
            )
            B_update[j, :] = res.x
            eta = np.tensordot(A_samples, res.x, axes=([1], [0]))
            gamma_update[j] = np.sum((Xj[:, None] - eta) ** 2) / (N * C)
        return B_update, gamma_update

    def csp_mstep(
        self,
        X,
        A_samples,
        B_start,
        dispersion,
        penalty_scale,
        temperature,
        *,
        response_cache,
        max_iter,
        tol,
        bound,
    ):
        return _solve_gaussian_weighted_l1(
            X,
            A_samples,
            B_start,
            dispersion,
            penalty_scale,
            temperature,
            x_sq_sum=response_cache,
            max_iter=max_iter,
            tol=tol,
            bound=bound,
        )

    def dispersion_distance_sq(self, old, new) -> float:
        return float(np.sum((np.asarray(old) - np.asarray(new)) ** 2))

    def conditional_log_prob(self, Y, eta, dispersion):
        Y = np.asarray(Y, dtype=float)
        eta = np.asarray(eta, dtype=float)
        gamma = np.asarray(dispersion, dtype=float).reshape(1, -1)
        return -0.5 * (
            np.log(2.0 * np.pi * gamma) + ((Y - eta) ** 2) / gamma
        )

    def sample_observed(self, eta, dispersion, rng):
        eta = np.asarray(eta, dtype=float)
        scale = np.sqrt(np.asarray(dispersion, dtype=float)).reshape(1, -1)
        return rng.normal(loc=eta, scale=scale, size=eta.shape)


NORMAL_FAMILY = NormalFamilyAdapter()

_FAMILY_REGISTRY = {
    alias: NORMAL_FAMILY for alias in NORMAL_FAMILY.aliases
}


def register_parametric_family(adapter):
    """Register one adapter under all of its aliases."""
    aliases = tuple(str(alias).strip().lower() for alias in adapter.aliases)
    if not aliases or any(not alias for alias in aliases):
        raise ValueError("A parametric family must define at least one alias.")
    for alias in aliases:
        existing = _FAMILY_REGISTRY.get(alias)
        if existing is not None and existing is not adapter:
            raise ValueError(f"Parametric family alias {alias!r} is already registered.")
    for alias in aliases:
        _FAMILY_REGISTRY[alias] = adapter
    return adapter


def resolve_parametric_family(value):
    """Return a registered family adapter from a name or adapter instance."""
    if not isinstance(value, (str, bytes)):
        required = (
            "aliases",
            "name",
            "has_dispersion",
            "validate_y",
            "validate_dispersion",
            "initial_working_response",
            "initialize",
            "observed_conditional_logit",
            "make_fixed_accumulator",
            "make_response_cache",
            "copy_dispersion",
            "fixed_mstep",
            "csp_mstep",
            "dispersion_distance_sq",
            "conditional_log_prob",
            "sample_observed",
        )
        if all(hasattr(value, attribute) for attribute in required):
            return value
        raise TypeError("family must be a registered name or family adapter.")
    normalized = str(value).strip().lower()
    if normalized in _FAMILY_REGISTRY:
        return _FAMILY_REGISTRY[normalized]
    raise NotImplementedError(
        "The current DDE interface implements family='Normal' only."
    )


__all__ = [
    "NORMAL_FAMILY",
    "NormalFamilyAdapter",
    "_ObservedNormalAccumulator",
    "_largest_eigenvalue_psd",
    "_normal_sample_statistics",
    "_soft_threshold",
    "_solve_gaussian_weighted_l1",
    "_solve_gaussian_weighted_l1_from_statistics",
    "_tlp_penalty",
    "_tlp_penalty_value_grad",
    "_update_deep_normal_observed",
    "register_parametric_family",
    "resolve_parametric_family",
]
