from lib_config import AppConfig
import numpy as np
from scipy.stats.qmc import LatinHypercube, scale
from typing import Optional, List, Tuple


class RandomSearch:
    def __init__(self, config: AppConfig):
        self.cfg = config
        runtime = getattr(config, "runtime", None)
        self.round_decimals = getattr(runtime, "round_decimals", 10)

    def _round_vector(self, values) -> np.ndarray:
        return np.round(np.asarray(values, dtype=float), decimals=self.round_decimals)

    def _sample_lhs_point(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        active_indices: Optional[List[int]] = None,
        fixed_point: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        dims = len(lower)

        if active_indices is None or len(active_indices) == dims:
            sampler = LatinHypercube(d=dims, seed=seed)
            sample = sampler.random(n=1)
            x_new = scale(sample, lower, upper)[0]
            return self._round_vector(x_new)

        if fixed_point is None:
            raise ValueError("fixed_point is required when active_indices is provided.")

        active = list(active_indices)
        x_new = self._round_vector(np.asarray(fixed_point, dtype=float).copy())

        sampler = LatinHypercube(d=len(active), seed=seed)
        sample = sampler.random(n=1)
        x_new[active] = scale(sample, lower[active], upper[active])[0]
        return self._round_vector(x_new)

    def search(
        self,
        history_data,
        param_names,
        lower_bounds,
        upper_bounds,
        objective_func=None,
        active_indices: Optional[List[int]] = None,
        fixed_point: Optional[np.ndarray] = None,
        routine_index: Optional[int] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, dict]:
        if objective_func is None:
            raise ValueError("RandomSearch requires objective_func.")

        lower = np.asarray(lower_bounds, dtype=float)
        upper = np.asarray(upper_bounds, dtype=float)

        seed = int(routine_index) if routine_index is not None else len(history_data)
        x_new = self._sample_lhs_point(
            lower=lower,
            upper=upper,
            active_indices=active_indices,
            fixed_point=fixed_point,
            seed=seed,
        )

        y_new, row = objective_func(param_names, x_new)
        evaluated_row = dict(row)
        evaluated_row["S11"] = float(np.round(y_new, self.round_decimals))

        return x_new, {
            "method": "random",
            "sampler": "lhs",
            "base_y": float(np.round(y_new, self.round_decimals)),
            "evaluated_rows": [evaluated_row],
            "n_evaluations": 1,
        }
