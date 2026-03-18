from lib_config import AppConfig
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Sequence, Optional, List, Tuple, Callable, Any

try:
    import torch
    from botorch.acquisition.analytic import ExpectedImprovement, UpperConfidenceBound
    from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood
except ImportError:
    torch = None
    ExpectedImprovement = None
    UpperConfidenceBound = None
    FixedFeatureAcquisitionFunction = None
    fit_gpytorch_mll = None
    SingleTaskGP = None
    Standardize = None
    optimize_acqf = None
    ExactMarginalLogLikelihood = None


try:
    import torch
    from botorch.acquisition.analytic import ExpectedImprovement, UpperConfidenceBound
    from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood
except ImportError:
    torch = None
    ExpectedImprovement = None
    UpperConfidenceBound = None
    FixedFeatureAcquisitionFunction = None
    fit_gpytorch_mll = None
    SingleTaskGP = None
    Standardize = None
    optimize_acqf = None
    ExactMarginalLogLikelihood = None


try:
    import torch
    from botorch.acquisition.analytic import ExpectedImprovement, UpperConfidenceBound
    from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.input import Normalize
    from botorch.models.transforms.outcome import Standardize
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood
except ImportError:
    torch = None
    ExpectedImprovement = None
    UpperConfidenceBound = None
    FixedFeatureAcquisitionFunction = None
    fit_gpytorch_mll = None
    SingleTaskGP = None
    Normalize = None
    Standardize = None
    optimize_acqf = None
    ExactMarginalLogLikelihood = None


try:
    import torch
    from botorch.acquisition.analytic import ExpectedImprovement, UpperConfidenceBound
    from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.input import Normalize
    from botorch.models.transforms.outcome import Standardize
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood
except ImportError:
    torch = None
    ExpectedImprovement = None
    UpperConfidenceBound = None
    FixedFeatureAcquisitionFunction = None
    fit_gpytorch_mll = None
    SingleTaskGP = None
    Normalize = None
    Standardize = None
    optimize_acqf = None
    ExactMarginalLogLikelihood = None


class GaussianProcess:
    def __init__(self, config: AppConfig,):
        self.cfg = config
        self.model = None
        self.mll = None
        self.train_X = None
        self.train_Y = None
        self.train_Y_model = None
        self.dtype = None
        self.device = None
        self.length_scale = None

    def _require_botorch(self) -> None:
        if torch is None or SingleTaskGP is None:
            raise ImportError(
                "BoTorch / GPyTorch / PyTorch are required. Install torch, gpytorch, and botorch first."
            )

    def _to_train_tensors(self, X_sample, y_sample):
        self._require_botorch()

        X_np = np.asarray(X_sample, dtype=np.float64)
        y_np = np.asarray(y_sample, dtype=np.float64).reshape(-1, 1)

        self.dtype = torch.double
        self.device = torch.device("cpu")

        train_X = torch.tensor(X_np, dtype=self.dtype, device=self.device)
        train_Y = torch.tensor(y_np, dtype=self.dtype, device=self.device)

        return train_X, train_Y


    def _get_input_bounds(self, dims: int):
        lower = np.asarray(self.cfg.hfss.lower_bounds, dtype=np.float64)
        upper = np.asarray(self.cfg.hfss.upper_bounds, dtype=np.float64)
        if len(lower) != dims or len(upper) != dims:
            raise ValueError("Config bounds do not match the training input dimension.")
        bounds = np.vstack([lower, upper])
        return torch.tensor(bounds, dtype=self.dtype, device=self.device)

    def _extract_length_scale(self) -> float:
        model = self.model
        covar_module = getattr(model, "covar_module", None)
        while covar_module is not None and hasattr(covar_module, "base_kernel"):
            covar_module = covar_module.base_kernel

        if covar_module is not None and hasattr(covar_module, "lengthscale"):
            ls = covar_module.lengthscale.detach().cpu().view(-1).double().numpy()
            return float(np.mean(ls))

        return float(self.cfg.opt.length_scale)

    def run_gp(self, X_sample, y_sample):
        train_X, train_Y = self._to_train_tensors(X_sample, y_sample)

        # Minimize S11 -> maximize negative S11 inside BoTorch.
        train_Y_model = -train_Y

        model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y_model,
            input_transform=Normalize(d=train_X.shape[-1], bounds=self._get_input_bounds(train_X.shape[-1])),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        model.eval()
        mll.eval()

        self.model = model
        self.mll = mll
        self.train_X = train_X
        self.train_Y = train_Y
        self.train_Y_model = train_Y_model
        self.length_scale = self._extract_length_scale()

    def _build_acquisition(self, acq_func, acq_params: dict):
        name = getattr(acq_func, "__name__", "") if acq_func is not None else ""
        params = dict(acq_params or {})

        if name == "expected_improvement" or params.get("name") == "expected_improvement":
            xi = float(params.get("xi", 0.01))
            best_f = self.train_Y_model.max() - xi
            return ExpectedImprovement(model=self.model, best_f=best_f, maximize=True)

        if name == "lower_confidence_bound" or params.get("name") == "lower_confidence_bound":
            kappa = float(params.get("kappa", 2.0))
            return UpperConfidenceBound(model=self.model, beta=kappa, maximize=True)

        raise ValueError(f"Unsupported acquisition function: {name or params.get('name')}")

    def optAcquisition(
        self,
        acq_func,
        X_sample,
        y_sample,
        lower_bounds,
        upper_bounds,
        acq_params,
        active_indices: Optional[List[int]] = None,
        fixed_point: Optional[np.ndarray] = None,
        n_restarts: int = 25,
    ) -> Tuple[np.ndarray, float]:
        self._require_botorch()

        lower_bounds = np.asarray(lower_bounds, dtype=float)
        upper_bounds = np.asarray(upper_bounds, dtype=float)
        dims = len(lower_bounds)
        acq = self._build_acquisition(acq_func, acq_params)

        # ---- full-dimensional optimization ----
        if active_indices is None or len(active_indices) == dims:
            bounds = torch.tensor(
                np.vstack([lower_bounds, upper_bounds]), dtype=self.dtype, device=self.device
            )
            best_x, best_acq_value = optimize_acqf(
                acq_function=acq,
                bounds=bounds,
                q=1,
                num_restarts=n_restarts,
                raw_samples=max(64, n_restarts * 8),
            )
            return (
                best_x.detach().cpu().view(-1).double().numpy(),
                float(best_acq_value.detach().cpu().view(-1)[0].item()),
            )

        # ---- partial optimization with fixed dims ----
        if fixed_point is None:
            raise ValueError("fixed_point is required when active_indices is provided.")

        active = list(active_indices)
        lb, ub = _sliceBounds(lower_bounds, upper_bounds, active)
        fixed_point = np.asarray(fixed_point, dtype=float).reshape(-1)

        columns = [i for i in range(dims) if i not in active]
        values = fixed_point[columns].tolist()
        acq_fixed = FixedFeatureAcquisitionFunction(
            acq_function=acq,
            d=dims,
            columns=columns,
            values=values,
        )

        bounds_free = torch.tensor(
            np.vstack([lb, ub]), dtype=self.dtype, device=self.device
        )
        best_z, best_acq_value = optimize_acqf(
            acq_function=acq_fixed,
            bounds=bounds_free,
            q=1,
            num_restarts=n_restarts,
            raw_samples=max(64, n_restarts * 8),
        )
        best_x = _reshapeX(fixed_point, active, best_z.detach().cpu().view(-1).double().numpy())
        return best_x, float(best_acq_value.detach().cpu().view(-1)[0].item())

    def search(
        self,
        history_data,
        param_names,
        lower_bounds,
        upper_bounds,
        objective_func=None,
        acq_func=None,
        acq_params=None,
        active_indices: Optional[List[int]] = None,
        fixed_point: Optional[np.ndarray] = None,
        n_restarts: int = 25,
        **kwargs,
    ) -> Tuple[np.ndarray, dict]:
        X_sample = np.asarray([[row[name] for name in param_names] for row in history_data], dtype=float)
        y_sample = np.asarray([[row["S11"]] for row in history_data], dtype=float)

        self.run_gp(X_sample, y_sample)

        if acq_func is None:
            acq_func = lower_confidence_bound
        if acq_params is None:
            acq_params = {"kappa": 2.0}

        x_new, acq_value = self.optAcquisition(
            acq_func=acq_func,
            X_sample=X_sample,
            y_sample=y_sample,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            acq_params=acq_params,
            active_indices=active_indices,
            fixed_point=fixed_point,
            n_restarts=n_restarts,
        )
        return x_new, {"acq": acq_value, "length_scale": self.length_scale}


# ==============================================================================
# Helper Functions kept for compatibility with existing plotting / notebooks
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
    if x1.ndim == 1:
        x1 = x1.reshape(1, -1)
    if x2.ndim == 1:
        x2 = x2.reshape(1, -1)
    dist = np.sqrt(np.sum((x1[:, np.newaxis, :] - x2[np.newaxis, :, :]) ** 2, axis=-1))

    if nu == 2.5:
        term1 = np.sqrt(5) * dist / length_scale
        term2 = 5 * dist**2 / (3 * length_scale**2)
        return (1 + term1 + term2) * np.exp(-term1)
    return rbf_kernel(x1, x2, length_scale)


def kernel(x1: np.ndarray, x2: np.ndarray, length_scale: float, KERNEL_TYPE='RBF') -> np.ndarray:
    if KERNEL_TYPE == 'RBF':
        return rbf_kernel(x1, x2, length_scale)
    if KERNEL_TYPE == 'Matern':
        return matern_kernel(x1, x2, length_scale, nu=2.5)
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
    if sigma == 0:
        return 0
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
    return -(mu - kappa * sigma)


def optimize_acquisition(X_sample, y_sample, Ky_opt_inv, length_scale, lower_bounds, upper_bounds, DIMS):
    best_acq_value = -np.inf
    best_x = None
    n_restarts = 25
    bounds = list(zip(lower_bounds, upper_bounds))

    for _ in range(n_restarts):
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
        best_x = np.random.uniform(lower_bounds, upper_bounds, DIMS)
    return best_x, best_acq_value


def negative_log_marginal_likelihood(params, X, y, noise_var):
    gamma = params[0]
    if gamma <= 0:
        return np.inf
    n = len(X)
    K = kernel(X, X, gamma)
    Ky = K + noise_var * np.identity(n) + 1e-6 * np.identity(n)
    try:
        L = np.linalg.cholesky(Ky)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        log_det_Ky = 2 * np.sum(np.log(np.diag(L)))
        return (0.5 * (y.T @ alpha) + 0.5 * log_det_Ky + 0.5 * n * np.log(2 * np.pi)).item()
    except np.linalg.LinAlgError:
        return np.inf
