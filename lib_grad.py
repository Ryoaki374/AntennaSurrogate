from lib_config import AppConfig
import numpy as np
from scipy.optimize import minimize
from typing import Optional, List, Tuple


class GradientSearch:
    def __init__(self, config: AppConfig):
        self.cfg = config

    def _build_base_point(self, best_x, active_indices, fixed_point):
        base_x = np.asarray(best_x, dtype=float).copy()
        if active_indices is None or fixed_point is None:
            return base_x
        x = np.asarray(fixed_point, dtype=float).copy()
        active = list(active_indices)
        x[active] = base_x[active]
        return x

    def search(
        self,
        history_data,
        param_names,
        lower_bounds,
        upper_bounds,
        objective_func=None,
        active_indices: Optional[List[int]] = None,
        fixed_point: Optional[np.ndarray] = None,
        maxiter: int = 20,
        **kwargs,
    ) -> Tuple[np.ndarray, dict]:
        if objective_func is None:
            raise ValueError("GradientSearch requires objective_func.")

        lower = np.asarray(lower_bounds, dtype=float)
        upper = np.asarray(upper_bounds, dtype=float)
        dims = len(lower)
        active = list(range(dims)) if active_indices is None else list(active_indices)

        best_row = min(history_data, key=lambda row: row["S11"])
        best_x = np.asarray([best_row[name] for name in param_names], dtype=float)
        base_x = self._build_base_point(best_x, active_indices, fixed_point)

        evaluated_rows = []
        eval_cache = {}

        def objective_active(z):
            x_full = base_x.copy()
            x_full[active] = np.asarray(z, dtype=float)
            x_full = np.clip(x_full, lower, upper)

            key = tuple(np.round(x_full, 12))
            if key in eval_cache:
                return eval_cache[key]

            y_value, row = objective_func(param_names, x_full)
            y_scalar = float(y_value)
            evaluated_rows.append(row)
            eval_cache[key] = y_scalar
            return y_scalar

        bounds_active = list(zip(lower[active], upper[active]))
        x0 = base_x[active]
        res = minimize(
            objective_active,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds_active,
            options={"maxiter": maxiter},
        )

        x_new = base_x.copy()
        x_new[active] = res.x
        x_new = np.clip(x_new, lower, upper)

        if active_indices is not None and fixed_point is not None:
            inactive = [i for i in range(dims) if i not in active]
            x_new[inactive] = np.asarray(fixed_point, dtype=float)[inactive]

        return x_new, {
            "method": "gradient",
            "solver": "L-BFGS-B",
            "base_y": float(res.fun),
            "evaluated_rows": evaluated_rows,
            "nit": int(getattr(res, "nit", 0)),
        }
