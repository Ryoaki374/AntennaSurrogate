from lib_config import AppConfig
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats.qmc import Sobol
from typing import Sequence, Optional, List, Tuple, Callable, Any


HORN_SIMPLEX_NAMES = ("f_wg", "f_t1", "f_mid", "f_t2", "f_ap")
DEFAULT_SOBOL_CANDIDATES = 4096


def _project_to_bounded_simplex(values, lower, upper, target=1.0, tol=1.0e-12, max_iter=100):
    """Euclidean projection onto ``sum(x)=target`` with box bounds."""
    values = np.asarray(values, dtype=float).reshape(-1)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)
    if values.shape != lower.shape or values.shape != upper.shape:
        raise ValueError("values, lower, and upper must have the same shape.")
    if np.any(lower > upper):
        raise ValueError("simplex lower bounds must not exceed upper bounds.")
    if lower.sum() - target > tol or target - upper.sum() > tol:
        raise ValueError("simplex bounds cannot satisfy the requested sum.")

    lo = float(np.min(values - upper))
    hi = float(np.max(values - lower))
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        projected = np.clip(values - mid, lower, upper)
        if projected.sum() > target:
            lo = mid
        else:
            hi = mid
    projected = np.clip(values - hi, lower, upper)
    residual = target - projected.sum()
    if abs(residual) > tol:
        free = np.where((projected > lower + tol) & (projected < upper - tol))[0]
        if free.size:
            projected[free] += residual / free.size
    return np.clip(projected, lower, upper)


def _bounded_simplex_from_unit(unit, lower, upper, target=1.0):
    """Map ``k-1`` unit coordinates to a bounded ``k``-part simplex.

    The inverse-Beta stick-breaking map produces a uniform Dirichlet sample for
    the mass remaining after the lower bounds are allocated.  Horn upper bounds
    are non-binding once the five 0.05 lower bounds and sum-to-one constraint are
    applied; unsupported binding upper bounds are rejected explicitly.
    """
    unit = np.asarray(unit, dtype=float)
    if unit.ndim == 1:
        unit = unit.reshape(1, -1)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)
    k = lower.size
    if unit.shape[1] != k - 1 or upper.size != k:
        raise ValueError("unit must have k-1 columns for a k-part simplex.")

    free_mass = float(target - lower.sum())
    if free_mass < 0.0:
        raise ValueError("simplex lower bounds exceed the requested sum.")
    if np.any(lower + free_mass > upper + 1.0e-12):
        raise ValueError("stick-breaking requires non-binding simplex upper bounds.")

    extras = np.zeros((unit.shape[0], k), dtype=float)
    remaining = np.full(unit.shape[0], free_mass, dtype=float)
    for j in range(k - 1):
        remaining_parts = k - j - 1
        u = np.clip(unit[:, j], 0.0, 1.0)
        fraction = 1.0 - np.power(1.0 - u, 1.0 / remaining_parts)
        extras[:, j] = remaining * fraction
        remaining -= extras[:, j]
    extras[:, -1] = remaining
    result = lower.reshape(1, -1) + extras
    # Remove accumulated floating-point residual from the dependent coordinate.
    result[:, -1] += target - result.sum(axis=1)
    return result

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


try:
    import torch
    from botorch.acquisition.analytic import ExpectedImprovement, UpperConfidenceBound
    from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.input import Normalize
    from botorch.models.transforms.outcome import Standardize
    from botorch.optim import optimize_acqf
    from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
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
    MaternKernel = None
    RBFKernel = None
    ScaleKernel = None
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
    from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
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
    MaternKernel = None
    RBFKernel = None
    ScaleKernel = None
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
    from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
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
    MaternKernel = None
    RBFKernel = None
    ScaleKernel = None
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
    from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
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
    MaternKernel = None
    RBFKernel = None
    ScaleKernel = None
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
    from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
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
    MaternKernel = None
    RBFKernel = None
    ScaleKernel = None
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
        self.external_dim = None
        self.simplex_indices = None
        self.external_lower_bounds = None
        self.external_upper_bounds = None

    def _configure_input_coordinates(self, param_names, lower_bounds, upper_bounds):
        """Use four independent coordinates for the five horn fractions."""
        names = list(param_names)
        lower = np.asarray(lower_bounds, dtype=float).reshape(-1)
        upper = np.asarray(upper_bounds, dtype=float).reshape(-1)
        if len(names) != lower.size or lower.shape != upper.shape:
            raise ValueError("parameter names and bounds must have matching lengths.")

        self.external_dim = len(names)
        self.external_lower_bounds = lower
        self.external_upper_bounds = upper
        if all(name in names for name in HORN_SIMPLEX_NAMES):
            indices = tuple(names.index(name) for name in HORN_SIMPLEX_NAMES)
            if indices != tuple(range(indices[0], indices[0] + len(indices))):
                raise ValueError("horn simplex parameters must be contiguous.")
            self.simplex_indices = indices
        else:
            self.simplex_indices = None

    def _encode_model_inputs(self, values):
        values = np.asarray(values, dtype=np.float64)
        was_vector = values.ndim == 1
        values = values.reshape(1, -1) if was_vector else values
        if self.external_dim is not None and values.shape[1] != self.external_dim:
            raise ValueError(
                f"Expected {self.external_dim} external input columns, got {values.shape[1]}."
            )
        if self.simplex_indices is not None:
            # f_ap is exactly determined by the other four fractions.
            values = np.delete(values, self.simplex_indices[-1], axis=1)
        return values.reshape(-1) if was_vector else values

    def _model_input_bounds(self):
        if self.external_lower_bounds is None or self.external_upper_bounds is None:
            lower = np.asarray(self.cfg.hfss.lower_bounds, dtype=np.float64)
            upper = np.asarray(self.cfg.hfss.upper_bounds, dtype=np.float64)
        else:
            lower = self.external_lower_bounds.copy()
            upper = self.external_upper_bounds.copy()

        if self.simplex_indices is not None:
            frac_lower = lower[list(self.simplex_indices)]
            frac_upper = upper[list(self.simplex_indices)]
            feasible_upper = 1.0 - (frac_lower.sum() - frac_lower)
            upper[list(self.simplex_indices)] = np.minimum(frac_upper, feasible_upper)
            lower = np.delete(lower, self.simplex_indices[-1])
            upper = np.delete(upper, self.simplex_indices[-1])
        return lower, upper

    def project_external_inputs(self, values):
        """Return inputs on the same feasible simplex used by HFSS."""
        values = np.asarray(values, dtype=float)
        was_vector = values.ndim == 1
        rows = values.reshape(1, -1).copy() if was_vector else values.copy()
        if self.simplex_indices is not None:
            idx = list(self.simplex_indices)
            lower = self.external_lower_bounds[idx]
            upper = self.external_upper_bounds[idx]
            for row in rows:
                row[idx] = _project_to_bounded_simplex(row[idx], lower, upper)
        return rows.reshape(-1) if was_vector else rows

    def sample_sobol_candidates(
        self,
        bounds,
        n_candidates,
        rng=None,
        active_indices=None,
        fixed_point=None,
    ):
        """Generate scrambled Sobol candidates, using four simplex coordinates."""
        bounds = _as_bounds_array(bounds)
        d = bounds.shape[0]
        active = list(range(d)) if active_indices is None else list(active_indices)
        if rng is None:
            rng = np.random.default_rng()
        seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))

        full_simplex_active = (
            self.simplex_indices is not None
            and all(index in active for index in self.simplex_indices)
        )
        simplex_active = set(active).intersection(self.simplex_indices or ())
        if simplex_active and not full_simplex_active:
            raise ValueError(
                "horn section fractions must be optimized together because one is dependent."
            )
        simplex_set = set(self.simplex_indices or ()) if full_simplex_active else set()
        ordinary_active = [index for index in active if index not in simplex_set]
        sobol_dim = len(ordinary_active) + (len(simplex_set) - 1 if simplex_set else 0)
        if sobol_dim <= 0:
            raise ValueError("at least one active design coordinate is required.")

        engine = Sobol(d=sobol_dim, scramble=True, seed=seed)
        power = int(np.ceil(np.log2(max(1, int(n_candidates)))))
        unit = engine.random_base2(power)[: int(n_candidates)]

        if len(active) == d:
            candidates = np.tile(bounds[:, 0], (len(unit), 1))
        else:
            if fixed_point is None:
                raise ValueError("fixed_point is required when active_indices is provided.")
            candidates = np.tile(np.asarray(fixed_point, dtype=float), (len(unit), 1))

        cursor = 0
        for index in ordinary_active:
            candidates[:, index] = (
                bounds[index, 0]
                + unit[:, cursor] * (bounds[index, 1] - bounds[index, 0])
            )
            cursor += 1

        if simplex_set:
            idx = list(self.simplex_indices)
            simplex_unit = unit[:, cursor : cursor + len(idx) - 1]
            candidates[:, idx] = _bounded_simplex_from_unit(
                simplex_unit,
                bounds[idx, 0],
                bounds[idx, 1],
            )
        elif self.simplex_indices is not None:
            candidates = self.project_external_inputs(candidates)
        return np.clip(candidates, bounds[:, 0], bounds[:, 1])

    def _optimize_simplex_acquisition_by_sobol(
        self,
        acq_name,
        acq_params,
        lower_bounds,
        upper_bounds,
        active_indices=None,
        fixed_point=None,
    ):
        params = dict(acq_params or {})
        rng = params.get("rng")
        if rng is None:
            rng = np.random.default_rng(params.get("seed"))
        n_candidates = int(params.get("n_candidates", DEFAULT_SOBOL_CANDIDATES))
        bounds = np.column_stack([lower_bounds, upper_bounds])
        candidates = self.sample_sobol_candidates(
            bounds,
            n_candidates,
            rng=rng,
            active_indices=active_indices,
            fixed_point=fixed_point,
        )
        mu, std = self.predict(candidates, return_std=True)
        mu = np.asarray(mu, dtype=float).reshape(-1)
        std = np.asarray(std, dtype=float).reshape(-1)

        if acq_name == "expected_improvement":
            xi = float(params.get("xi", 0.01))
            improvement = float(np.min(self.train_Y.detach().cpu().numpy())) - mu - xi
            safe_std = np.maximum(std, 1.0e-15)
            z_score = improvement / safe_std
            values = improvement * norm.cdf(z_score) + safe_std * norm.pdf(z_score)
            values = np.where(std > 0.0, values, np.maximum(improvement, 0.0))
            best_idx = int(np.argmax(values))
            return candidates[best_idx], float(values[best_idx])

        kappa = float(params.get("kappa", 2.0))
        lcb = mu - kappa * std
        best_idx = int(np.argmin(lcb))
        return candidates[best_idx], float(-lcb[best_idx])

    def _require_botorch(self) -> None:
        if torch is None or SingleTaskGP is None:
            raise ImportError(
                "BoTorch / GPyTorch / PyTorch are required. Install torch, gpytorch, and botorch first."
            )

    def _to_train_tensors(self, X_sample, y_sample):
        self._require_botorch()

        X_np = self._encode_model_inputs(X_sample)
        y_np = np.asarray(y_sample, dtype=np.float64).reshape(-1, 1)

        self.dtype = torch.double
        self.device = torch.device("cpu")

        train_X = torch.tensor(X_np, dtype=self.dtype, device=self.device)
        train_Y = torch.tensor(y_np, dtype=self.dtype, device=self.device)

        return train_X, train_Y


    def _get_input_bounds(self, dims: int):
        lower, upper = self._model_input_bounds()
        if len(lower) != dims or len(upper) != dims:
            raise ValueError("Config bounds do not match the training input dimension.")
        bounds = np.vstack([lower, upper])
        return torch.tensor(bounds, dtype=self.dtype, device=self.device)


    def _build_covar_module(self, dims: int):
        kernel_type = str(self.cfg.opt.kernel_type).lower()
        if kernel_type == "rbf":
            base_kernel = RBFKernel(ard_num_dims=dims)
        elif kernel_type in {"matern", "matern52", "matern_5_2", "matern5/2"}:
            base_kernel = MaternKernel(nu=2.5, ard_num_dims=dims)
        else:
            raise ValueError(f"Unsupported kernel_type: {self.cfg.opt.kernel_type}")
        base_kernel.initialize(lengthscale=float(self.cfg.opt.length_scale))
        return ScaleKernel(base_kernel)

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

        # Minimize the configured objective -> maximize its negative inside BoTorch.
        train_Y_model = -train_Y

        model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y_model,
            covar_module=self._build_covar_module(train_X.shape[-1]),
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

    def fit(self, X_train, y_train):
        """Fit the GP using the sklearn-like API expected by LCB helpers."""
        self.run_gp(X_train, y_train)
        return self

    def predict(self, X, return_std: bool = False, return_cov: bool = False):
        """Predict posterior quantities on the original minimization scale.

        The internal BoTorch model is trained on ``-y`` so that existing BoTorch
        acquisitions can maximize improvement.  This method converts the mean
        back to the original objective scale while keeping variances/covariances
        unchanged.
        """
        self._require_botorch()
        if self.model is None:
            raise RuntimeError("GaussianProcess must be fit before calling predict.")
        if return_std and return_cov:
            raise ValueError("Only one of return_std and return_cov may be True.")

        X_np = np.asarray(X, dtype=np.float64)
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)
        X_model = self._encode_model_inputs(X_np)
        X_tensor = torch.tensor(X_model, dtype=self.dtype, device=self.device)

        with torch.no_grad():
            posterior = self.model.posterior(X_tensor)
            mean_model = posterior.mean.detach().cpu().view(-1).double().numpy()
            mu = -mean_model

            if return_cov:
                cov = posterior.mvn.covariance_matrix.detach().cpu().double().numpy()
                cov = np.asarray(cov, dtype=float).reshape(X_model.shape[0], X_model.shape[0])
                cov = 0.5 * (cov + cov.T)
                return mu, cov

            if return_std:
                var = posterior.variance.detach().cpu().view(-1).double().numpy()
                std = np.sqrt(np.maximum(var, 0.0))
                return mu, std

        return mu

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
        default_objective_col = getattr(getattr(self.cfg, "objective", None), "name", "Objective")
        objective_col = kwargs.get("objective_col", default_objective_col)
        self._configure_input_coordinates(param_names, lower_bounds, upper_bounds)
        X_sample = np.asarray([[row[name] for name in param_names] for row in history_data], dtype=float)
        y_sample = np.asarray([[row[objective_col]] for row in history_data], dtype=float)

        self.run_gp(X_sample, y_sample)

        if acq_func is None:
            acq_func = lower_confidence_bound
        if acq_params is None:
            acq_params = {"kappa": 2.0}

        params = dict(acq_params or {})
        acq_name = getattr(acq_func, "__name__", "") if acq_func is not None else ""
        acq_name = str(params.get("name", acq_name)).lower()
        robust_lcb_names = {
            "robust_lcb_on_j",
            "robust_lcb",
            "rlcb",
            "robust_lcb_penalty",
            "robust_lcb_penalty_on_j",
            "rlcb_penalty",
            "rlcbp",
        }
        if acq_name in robust_lcb_names:
            bounds = np.vstack([np.asarray(lower_bounds, dtype=float), np.asarray(upper_bounds, dtype=float)]).T
            d = bounds.shape[0]
            Sigma = params.get("Sigma", params.get("sigma", None))
            if Sigma is None:
                perturbation_std = np.asarray(
                    params.get("perturbation_std", 0.01 * (bounds[:, 1] - bounds[:, 0])),
                    dtype=float,
                )
                if perturbation_std.ndim == 0:
                    perturbation_std = np.full(d, float(perturbation_std))
                Sigma = np.diag(perturbation_std ** 2)

            rng = params.get("rng", None)
            if rng is None:
                rng = np.random.default_rng(params.get("seed", None))

            penalty_names = {
                "robust_lcb_penalty",
                "robust_lcb_penalty_on_j",
                "rlcb_penalty",
                "rlcbp",
            }
            optimizer = (
                optimize_robust_lcb_penalty_by_sobol_search
                if acq_name in penalty_names
                else optimize_robust_lcb_by_sobol_search
            )
            optimizer_kwargs = {
                "gp": self,
                "Sigma": Sigma,
                "bounds": bounds,
                "rng": rng,
                "n_perturb": int(params.get("n_perturb", 64)),
                "kappa": float(params.get("kappa", 1.0)),
                "n_candidates": params.get("n_candidates", None),
                "active_indices": active_indices,
                "fixed_point": fixed_point,
            }
            if acq_name in penalty_names:
                optimizer_kwargs["penalty_weight"] = float(params.get("penalty_weight", 0.0))

            # robust LCB on J(x), optionally with a posterior-mean variation penalty.
            x_new, acq_value = optimizer(**optimizer_kwargs)

            # standard LCB random-search alternative for direct comparison:
            # x_new, acq_value = optimize_lcb_by_random_search(
            #     gp=self,
            #     bounds=bounds,
            #     rng=rng,
            #     kappa=float(params.get("kappa", 1.0)),
            #     active_indices=active_indices,
            #     fixed_point=fixed_point,
            # )
            return x_new, {"acq": acq_value, "length_scale": self.length_scale, "method": acq_name}

        # standard LCB / EI BoTorch path
        if self.simplex_indices is not None:
            standard_name = str(params.get("name", acq_name)).lower()
            if standard_name not in {"expected_improvement", "lower_confidence_bound"}:
                raise ValueError(f"Unsupported simplex acquisition function: {standard_name}")
            x_new, acq_value = self._optimize_simplex_acquisition_by_sobol(
                standard_name,
                params,
                lower_bounds,
                upper_bounds,
                active_indices=active_indices,
                fixed_point=fixed_point,
            )
            return x_new, {
                "acq": acq_value,
                "length_scale": self.length_scale,
                "method": f"{standard_name}_sobol",
            }

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



def _as_bounds_array(bounds) -> np.ndarray:
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2:
        raise ValueError("bounds must have shape (d, 2) or (2, d).")
    if bounds.shape[1] == 2:
        return bounds
    if bounds.shape[0] == 2:
        return bounds.T
    raise ValueError("bounds must have shape (d, 2) or (2, d).")


def sample_input_perturbations(x, Sigma, n_samples, bounds, rng):
    """Sample clipped Gaussian input perturbation points around ``x``.

    Returns ``clip(x + delta, bounds)`` with shape ``(n_samples, d)``.
    """
    if rng is None:
        rng = np.random.default_rng()
    x = np.asarray(x, dtype=float).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=float)
    bounds = _as_bounds_array(bounds)
    d = x.size
    if Sigma.shape != (d, d):
        raise ValueError("Sigma must have shape (d, d).")
    if bounds.shape != (d, 2):
        raise ValueError("bounds must match x dimension.")

    deltas = rng.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=int(n_samples))
    Z = x.reshape(1, -1) + deltas
    return np.clip(Z, bounds[:, 0], bounds[:, 1])


def robust_lcb_on_J(
    x,
    gp,
    Sigma,
    bounds,
    n_perturb=64,
    kappa=1.0,
    rng=None,
    perturbations=None,
    eps=1e-12,
    return_details=False,
):
    """Compute robust LCB for J(x)=E_delta[f(x+delta)] using GP covariance."""
    if rng is None:
        rng = np.random.default_rng()
    x = np.asarray(x, dtype=float).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=float)
    bounds = _as_bounds_array(bounds)
    M = int(n_perturb)
    if M <= 0:
        raise ValueError("n_perturb must be positive.")

    if perturbations is None:
        Z = sample_input_perturbations(x, Sigma, M, bounds, rng)
    else:
        perturbations = np.asarray(perturbations, dtype=float)
        if perturbations.ndim != 2 or perturbations.shape[1] != x.size:
            raise ValueError("perturbations must have shape (n_perturb, d).")
        M = perturbations.shape[0]
        Z = np.clip(x.reshape(1, -1) + perturbations, bounds[:, 0], bounds[:, 1])

    projector = getattr(gp, "project_external_inputs", None)
    if callable(projector):
        Z = projector(Z)

    mu_Z, cov_Z = gp.predict(Z, return_cov=True)
    mu_Z = np.asarray(mu_Z, dtype=float).reshape(-1)
    cov_Z = np.asarray(cov_Z, dtype=float).reshape(M, M)
    cov_Z = 0.5 * (cov_Z + cov_Z.T)

    mu_J = float(np.mean(mu_Z))
    var_J = float(np.sum(cov_Z) / (M ** 2))
    var_J = max(var_J, float(eps))
    sigma_J = float(np.sqrt(var_J))
    lcb_J = float(mu_J - float(kappa) * sigma_J)
    if M <= 1:
        posterior_mu_sample_std = 0.0
    else:
        posterior_mu_sample_std = float(
            np.sqrt(np.sum((mu_Z - mu_J) ** 2) / (M - 1))
        )

    if not return_details:
        return lcb_J
    return {
        "lcb": lcb_J,
        "mu_J": mu_J,
        "sigma_J": sigma_J,
        "var_J": var_J,
        "posterior_mu_sample_std": posterior_mu_sample_std,
        "Z": Z,
        "mu_Z": mu_Z,
    }


def robust_lcb_penalty_on_J(
    x,
    gp,
    Sigma,
    bounds,
    n_perturb=64,
    kappa=1.0,
    penalty_weight=0.0,
    rng=None,
    perturbations=None,
    eps=1e-12,
    return_details=False,
):
    """Compute robust LCB on J(x) plus posterior mean sample-std penalty."""
    details = robust_lcb_on_J(
        x,
        gp,
        Sigma,
        bounds,
        n_perturb=n_perturb,
        kappa=kappa,
        rng=rng,
        perturbations=perturbations,
        eps=eps,
        return_details=True,
    )
    penalty_value = float(
        details["mu_J"]
        - float(kappa) * details["sigma_J"]
        + float(penalty_weight) * details["posterior_mu_sample_std"]
    )
    if not return_details:
        return penalty_value

    details = dict(details)
    details["lcb_penalty"] = penalty_value
    details["penalty_weight"] = float(penalty_weight)
    return details


def optimize_lcb_by_random_search(
    gp,
    bounds,
    rng,
    kappa=1.0,
    n_candidates=None,
    active_indices: Optional[List[int]] = None,
    fixed_point: Optional[np.ndarray] = None,
):
    """Minimize the standard LCB, mu(x) - kappa * sigma(x), by random search."""
    if rng is None:
        rng = np.random.default_rng()
    bounds = _as_bounds_array(bounds)
    d = bounds.shape[0]
    active = list(range(d)) if active_indices is None else list(active_indices)
    if n_candidates is None:
        n_candidates = max(256, 64 * len(active))

    if len(active) == d:
        X_cand = rng.uniform(bounds[:, 0], bounds[:, 1], size=(int(n_candidates), d))
    else:
        if fixed_point is None:
            raise ValueError("fixed_point is required when active_indices is provided.")
        fixed_point = np.asarray(fixed_point, dtype=float).reshape(-1)
        X_cand = np.tile(fixed_point, (int(n_candidates), 1))
        X_cand[:, active] = rng.uniform(
            bounds[active, 0],
            bounds[active, 1],
            size=(int(n_candidates), len(active)),
        )
        X_cand = np.clip(X_cand, bounds[:, 0], bounds[:, 1])

    mu, std = gp.predict(X_cand, return_std=True)
    values = np.asarray(mu, dtype=float).reshape(-1) - float(kappa) * np.asarray(std, dtype=float).reshape(-1)
    best_idx = int(np.argmin(values))
    return X_cand[best_idx], float(values[best_idx])


def optimize_robust_lcb_by_sobol_search(
    gp,
    Sigma,
    bounds,
    rng,
    n_perturb=64,
    kappa=1.0,
    n_candidates=None,
    active_indices: Optional[List[int]] = None,
    fixed_point: Optional[np.ndarray] = None,
):
    """Minimize robust LCB on J(x) over scrambled Sobol candidates."""
    if rng is None:
        rng = np.random.default_rng()
    Sigma = np.asarray(Sigma, dtype=float)
    bounds = _as_bounds_array(bounds)
    d = bounds.shape[0]
    active = list(range(d)) if active_indices is None else list(active_indices)
    if Sigma.shape != (d, d):
        raise ValueError("Sigma must have shape (d, d).")
    if n_candidates is None:
        n_candidates = DEFAULT_SOBOL_CANDIDATES

    sampler = getattr(gp, "sample_sobol_candidates", None)
    if callable(sampler):
        X_cand = sampler(
            bounds,
            n_candidates,
            rng=rng,
            active_indices=active_indices,
            fixed_point=fixed_point,
        )
    else:
        seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        engine = Sobol(d=len(active), scramble=True, seed=seed)
        power = int(np.ceil(np.log2(max(1, int(n_candidates)))))
        unit = engine.random_base2(power)[: int(n_candidates)]
        if len(active) == d:
            X_cand = bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])
        else:
            if fixed_point is None:
                raise ValueError("fixed_point is required when active_indices is provided.")
            X_cand = np.tile(np.asarray(fixed_point, dtype=float), (len(unit), 1))
            X_cand[:, active] = (
                bounds[active, 0] + unit * (bounds[active, 1] - bounds[active, 0])
            )

    L_sigma = np.linalg.cholesky(Sigma + 1e-12 * np.eye(d))
    standard_normals = rng.normal(size=(int(n_perturb), d))
    perturbations = standard_normals @ L_sigma.T

    values = np.array([
        robust_lcb_on_J(
            x,
            gp,
            Sigma,
            bounds,
            n_perturb=n_perturb,
            kappa=kappa,
            perturbations=perturbations,
        )
        for x in X_cand
    ], dtype=float)
    best_idx = int(np.argmin(values))
    return X_cand[best_idx], float(values[best_idx])


def optimize_robust_lcb_penalty_by_sobol_search(
    gp,
    Sigma,
    bounds,
    rng,
    n_perturb=64,
    kappa=1.0,
    penalty_weight=0.0,
    n_candidates=None,
    active_indices: Optional[List[int]] = None,
    fixed_point: Optional[np.ndarray] = None,
):
    """Minimize penalized robust LCB over scrambled Sobol candidates."""
    if rng is None:
        rng = np.random.default_rng()
    Sigma = np.asarray(Sigma, dtype=float)
    bounds = _as_bounds_array(bounds)
    d = bounds.shape[0]
    active = list(range(d)) if active_indices is None else list(active_indices)
    if Sigma.shape != (d, d):
        raise ValueError("Sigma must have shape (d, d).")
    if n_candidates is None:
        n_candidates = DEFAULT_SOBOL_CANDIDATES

    sampler = getattr(gp, "sample_sobol_candidates", None)
    if callable(sampler):
        X_cand = sampler(
            bounds,
            n_candidates,
            rng=rng,
            active_indices=active_indices,
            fixed_point=fixed_point,
        )
    else:
        seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        engine = Sobol(d=len(active), scramble=True, seed=seed)
        power = int(np.ceil(np.log2(max(1, int(n_candidates)))))
        unit = engine.random_base2(power)[: int(n_candidates)]
        if len(active) == d:
            X_cand = bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])
        else:
            if fixed_point is None:
                raise ValueError("fixed_point is required when active_indices is provided.")
            X_cand = np.tile(np.asarray(fixed_point, dtype=float), (len(unit), 1))
            X_cand[:, active] = (
                bounds[active, 0] + unit * (bounds[active, 1] - bounds[active, 0])
            )

    L_sigma = np.linalg.cholesky(Sigma + 1e-12 * np.eye(d))
    standard_normals = rng.normal(size=(int(n_perturb), d))
    perturbations = standard_normals @ L_sigma.T

    values = np.array([
        robust_lcb_penalty_on_J(
            x,
            gp,
            Sigma,
            bounds,
            n_perturb=n_perturb,
            kappa=kappa,
            penalty_weight=penalty_weight,
            perturbations=perturbations,
        )
        for x in X_cand
    ], dtype=float)
    best_idx = int(np.argmin(values))
    return X_cand[best_idx], float(values[best_idx])


# Backward-compatible names for callers outside this repository.
optimize_robust_lcb_by_random_search = optimize_robust_lcb_by_sobol_search
optimize_robust_lcb_penalty_by_random_search = optimize_robust_lcb_penalty_by_sobol_search


def robust_validation_summary(candidates, gp, Sigma, bounds, rng=None, n_perturb=64, eps=1e-12):
    """Summarize nominal and robust posterior quantities for candidate points."""
    if rng is None:
        rng = np.random.default_rng()
    summary = {}
    for name, x in candidates.items():
        x_arr = np.asarray(x, dtype=float).reshape(1, -1)
        nominal_mu, nominal_std = gp.predict(x_arr, return_std=True)
        details = robust_lcb_on_J(
            x_arr.reshape(-1),
            gp,
            Sigma,
            bounds,
            n_perturb=n_perturb,
            kappa=0.0,
            rng=rng,
            eps=eps,
            return_details=True,
        )
        summary[name] = {
            "nominal_posterior_mean": float(np.asarray(nominal_mu).reshape(-1)[0]),
            "nominal_posterior_std": float(np.asarray(nominal_std).reshape(-1)[0]),
            "posterior_robust_mean": details["mu_J"],
            "posterior_robust_std": details["sigma_J"],
            "posterior_mu_sample_std": details["posterior_mu_sample_std"],
        }
    if summary:
        robust_best_name = min(summary, key=lambda key: summary[key]["posterior_robust_mean"])
        summary["robust_best"] = {
            "name": robust_best_name,
            "posterior_robust_mean": summary[robust_best_name]["posterior_robust_mean"],
        }
    return summary

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

