from lib_config import AppConfig
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Sequence, Optional, List, Tuple, Callable

class GaussianProcess:
    def __init__(self, config: AppConfig,):
        self.cfg = config

    def run_gp(self, X_sample, y_sample, current_gamma):
        res_hyper = minimize(
            fun= negative_log_marginal_likelihood, x0=[current_gamma],
            args=(X_sample, y_sample, self.cfg.opt.noise_var), bounds=[(0.1, 1e2)]
        )
        optimized_length_scale = res_hyper.x[0]

        K_opt = kernel(X_sample, X_sample, optimized_length_scale)
        Ky_opt = K_opt + self.cfg.opt.noise_var * np.identity(len(X_sample)) + 1e-6 * np.identity(len(X_sample))
        Ky_opt_inv = np.linalg.inv(Ky_opt)
        return optimized_length_scale, K_opt, Ky_opt, Ky_opt_inv, res_hyper
    
    
    def optAcquisition(self, acq_func, X_sample, y_sample, Ky_inv, gamma, lower_bounds, upper_bounds, acq_params):

        dims = len(lower_bounds)
        best_acq_value = -np.inf
        best_x = None
        n_restarts = 25
        bounds = list(zip(lower_bounds, upper_bounds))

        def acquisitionWrapper(x):
            val = acq_func(
                x, X_sample, y_sample, Ky_inv, gamma, **acq_params
            )
            return -val # minimize(-acq) = maximize(acq)
        
        for i in range(n_restarts):
            x0 = np.random.uniform(lower_bounds, upper_bounds, dims)
            res = minimize(
                fun=acquisitionWrapper, x0=x0, bounds=bounds, method="L-BFGS-B"
            )
            if res.success and -res.fun > best_acq_value:
                best_acq_value = -res.fun
                best_x = res.x

        if best_x is None:
            print("  > WARNING: Acquisition optimization failed. Using a random point.")
            best_x = np.random.uniform(lower_bounds, upper_bounds, dims)
    
        return best_x, best_acq_value
    
    def optAcquisition(self, acq_func, X_sample, y_sample, Ky_inv, gamma, lower_bounds, upper_bounds,acq_params, active_indices: Optional[List[int]] = None, fixed_point: Optional[np.ndarray] = None,n_restarts: int = 25,) -> Tuple[np.ndarray, float]:

        lower_bounds = np.asarray(lower_bounds, dtype=float)
        upper_bounds = np.asarray(upper_bounds, dtype=float)
        dims = len(lower_bounds)

        # ---- full-dimensional optimization ----
        if active_indices is None or len(active_indices) == dims:
            bounds_full = list(zip(lower_bounds.tolist(), upper_bounds.tolist()))

            def acquisitionWrapper(x):
                val = acq_func(x, X_sample, y_sample, Ky_inv, gamma, **acq_params)
                return -val

            best_x, best_acq_value = _optimizer(
                fun=acquisitionWrapper,
                bounds=bounds_full,
                x0_sampler=lambda: np.random.uniform(lower_bounds, upper_bounds, size=dims),
                n_restarts=n_restarts,
            )

            if best_x is None:
                print("  > WARNING: Acquisition optimization failed. Using a random point.")
                best_x = np.random.uniform(lower_bounds, upper_bounds, size=dims)
                best_acq_value = acq_func(best_x, X_sample, y_sample, Ky_inv, gamma, **acq_params)

            return best_x, best_acq_value

        # ---- partial optimization with fixed dims ----
        if fixed_point is None:
            raise ValueError("fixed_point is required when active_indices is provided.")

        active = list(active_indices)
        lb, ub = _sliceBounds(lower_bounds, upper_bounds, active)
        bounds_free = list(zip(lb.tolist(), ub.tolist()))

        fixed_point = np.asarray(fixed_point, dtype=float)
        if fixed_point.shape != (dims,):
            fixed_point = fixed_point.reshape(-1)

        def acquisitionWrapperFree(z):
            x = _reshapeX(fixed_point, active, z)
            val = acq_func(x, X_sample, y_sample, Ky_inv, gamma, **acq_params)
            return -val

        best_z, best_acq_value = _optimizer(
            fun=acquisitionWrapperFree,
            bounds=bounds_free,
            x0_sampler=lambda: np.random.uniform(lb, ub, size=len(active)),
            n_restarts=n_restarts,
        )

        if best_z is None:
            print("  > WARNING: Acquisition optimization failed. Using a random point (with fixed dims).")
            z = np.random.uniform(lb, ub, size=len(active))
            best_x = _reshapeX(fixed_point, active, z)
            best_acq_value = acq_func(best_x, X_sample, y_sample, Ky_inv, gamma, **acq_params)
            return best_x, best_acq_value

        best_x = _reshapeX(fixed_point, active, best_z)
        return best_x, best_acq_value



# ==============================================================================
# 2. Gaussian Process Helper Functions
# ==============================================================================
def _reshapeX(fixed_point: np.ndarray, active_indices: Sequence[int], z: np.ndarray) -> np.ndarray:
    x = np.array(fixed_point, dtype=float, copy=True)
    x[list(active_indices)] = np.asarray(z, dtype=float)
    return x

def _sliceBounds(lower_bounds, upper_bounds, active_indices: Sequence[int],) -> Tuple[np.ndarray, np.ndarray]:
        lb = np.asarray(lower_bounds, dtype=float)[list(active_indices)]
        ub = np.asarray(upper_bounds, dtype=float)[list(active_indices)]
        return lb, ub

def _optimizer(fun: Callable[[np.ndarray], float], bounds: List[Tuple[float, float]], x0_sampler: Callable[[], np.ndarray], n_restarts: int,) -> Tuple[Optional[np.ndarray], float]:
    best_val = -np.inf
    best_x = None
    for _ in range(n_restarts):
        x0 = x0_sampler()
        res = minimize(fun=fun, x0=x0, bounds=bounds, method="L-BFGS-B")
        if res.success:
            cand_val = -res.fun
            if cand_val > best_val:
                best_val = cand_val
                best_x = res.x
    return best_x, best_val


def rbf_kernel(x1: np.ndarray, x2: np.ndarray, gamma: float) -> np.ndarray:
    sqdist = np.sum(x1**2, 1).reshape(-1, 1) - 2 * np.dot(x1, x2.T) + np.sum(x2**2, 1)
    return np.exp(-gamma * sqdist)

def matern_kernel(x1: np.ndarray, x2: np.ndarray, length_scale: float = 1.0, nu: float = 2.5) -> np.ndarray:
    if x1.ndim == 1: x1 = x1.reshape(1, -1)
    if x2.ndim == 1: x2 = x2.reshape(1, -1)
    dist = np.sqrt(np.sum((x1[:, np.newaxis, :] - x2[np.newaxis, :, :]) ** 2, axis=-1))
    
    if nu == 2.5:
        term1 = np.sqrt(5) * dist / length_scale
        term2 = 5 * dist**2 / (3 * length_scale**2)
        return (1 + term1 + term2) * np.exp(-term1)
    else:
        return rbf_kernel(x1, x2, length_scale)

def kernel(x1: np.ndarray, x2: np.ndarray, length_scale: float, KERNEL_TYPE='RBF') -> np.ndarray:
    if KERNEL_TYPE == 'RBF':
        return rbf_kernel(x1, x2, length_scale)
    elif KERNEL_TYPE == 'Matern':
        return matern_kernel(x1, x2, length_scale, nu=2.5)
    else:
        raise ValueError("Unknown KERNEL_TYPE specified.")

def get_posterior(x_new, X_sample, y_sample, Ky_opt_inv, length_scale):
    x_new = x_new.reshape(1, -1)
    K_star = kernel(X_sample, x_new, length_scale)
    mu_post = K_star.T @ Ky_opt_inv @ y_sample
    K_star_star = 1.0 
    cov_post = K_star_star - K_star.T @ Ky_opt_inv @ K_star
    s2_post = np.maximum(0, cov_post.item())
    return mu_post.item(), np.sqrt(s2_post)

def expected_improvement(x_new, X_sample, y_sample, Ky_opt_inv, length_scale, xi=0.01):
    mu, sigma = get_posterior(x_new, X_sample, y_sample, Ky_opt_inv, length_scale)
    y_best = np.min(y_sample)
    if sigma == 0: return 0
    imp = y_best - mu - xi
    Z = imp / sigma
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    return ei

def negative_expected_improvement(x_new, X_sample, y_sample, Ky_opt_inv, length_scale, xi=0.01):
    return -expected_improvement(x_new, X_sample, y_sample, Ky_opt_inv, length_scale, xi)

def lower_confidence_bound(
    x_new, X_sample, y_sample, Ky_opt_inv, length_scale, kappa=2.0
):
    mu, sigma = get_posterior(
        x_new, X_sample, y_sample, Ky_opt_inv, length_scale
    )
    # LCB wants small mu - kappa*sigma.
    # We return NEGATIVE LCB because optAcquisition minimizes the return value.
    return -(mu - kappa * sigma)

def optimize_acquisition(X_sample, y_sample, Ky_opt_inv, length_scale, lower_bounds, upper_bounds, DIMS):
    best_acq_value = -np.inf
    best_x = None
    n_restarts = 25
    bounds = list(zip(lower_bounds, upper_bounds))

    for i in range(n_restarts):
        x0 = np.random.uniform(lower_bounds, upper_bounds, DIMS)
        res = minimize(
            fun=negative_expected_improvement, x0=x0,
            args=(X_sample, y_sample, Ky_opt_inv, length_scale),
            bounds=bounds, method='L-BFGS-B'
        )
        if res.success and -res.fun > best_acq_value:
            best_acq_value = -res.fun
            best_x = res.x
    
    if best_x is None:
        print("  > WARNING: Acquisition optimization failed. Using a random point.")
        best_x = np.random.uniform(lower_bounds, upper_bounds, DIMS)
    return best_x, best_acq_value

def negative_log_marginal_likelihood(params, X, y, noise_var):
    gamma = params[0]
    if gamma <= 0: return np.inf
    n = len(X)
    K = kernel(X, X, gamma)
    #Ky = K + noise_var * np.identity(n) 
    Ky = K + noise_var * np.identity(n) + 1e-6 * np.identity(n)
    try:
        L = np.linalg.cholesky(Ky)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        log_det_Ky = 2 * np.sum(np.log(np.diag(L)))
        return (0.5 * (y.T @ alpha) + 0.5 * log_det_Ky + 0.5 * n * np.log(2 * np.pi)).item()
    except np.linalg.LinAlgError:
        return np.inf