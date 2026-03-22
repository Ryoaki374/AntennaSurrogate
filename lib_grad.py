from lib_config import AppConfig
import numpy as np
from scipy.optimize import minimize
from typing import Optional, List, Tuple


def _round_vector(values: np.ndarray, decimals: int = 2) -> np.ndarray:
    return np.round(np.asarray(values, dtype=float), decimals=decimals)


def _format_vector(values: np.ndarray) -> str:
    return np.array2string(_round_vector(values, decimals=2), precision=2, separator=", ")


class GradientSearch:
    def __init__(self, config: AppConfig):
        self.cfg = config

    def _build_base_point(self, best_x, active_indices, fixed_point):
        base_x = np.asarray(best_x, dtype=float).copy()
        if active_indices is None or fixed_point is None:
            return base_x
        x = _round_vector(np.asarray(fixed_point, dtype=float).copy(), decimals=2)
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
        maxfun: Optional[int] = None,
        start_row: Optional[dict] = None,
        routine_index: Optional[int] = None,
        routine_total: Optional[int] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, dict]:
        if objective_func is None:
            raise ValueError("GradientSearch requires objective_func.")

        lower = np.asarray(lower_bounds, dtype=float)
        upper = np.asarray(upper_bounds, dtype=float)
        dims = len(lower)
        active = list(range(dims)) if active_indices is None else list(active_indices)

        best_row = dict(start_row) if start_row is not None else min(history_data, key=lambda row: row["S11"])
        best_x = _round_vector(np.asarray([best_row[name] for name in param_names], dtype=float), decimals=2)
        base_x = self._build_base_point(best_x, active_indices, fixed_point)

        if routine_index is not None and routine_total is not None:
            remaining_before = routine_total - routine_index + 1
            remaining_after = routine_total - routine_index
            print(
                f"[grad_lbfgs] start routine {routine_index}/{routine_total} "
                f"(remaining incl. this: {remaining_before}, remaining after: {remaining_after})"
            )

        evaluated_rows = []
        eval_cache = {}
        eval_count = 0

        def objective_active(z):
            nonlocal eval_count
            x_full = base_x.copy()
            x_full[active] = np.asarray(z, dtype=float)
            x_full = np.clip(x_full, lower, upper)
            x_full = _round_vector(x_full, decimals=2)

            key = tuple(x_full.tolist())
            if key in eval_cache:
                cached_value = eval_cache[key]
                print(
                    f"[grad_lbfgs] routine {routine_index or '-'} cache hit "
                    f"f={cached_value:.6f} x={_format_vector(x_full)}"
                )
                return cached_value

            y_value, row = objective_func(param_names, x_full)
            y_scalar = float(y_value)
            eval_count += 1
            debug_row = dict(row)
            debug_row["debug_eval_idx"] = eval_count
            debug_row["debug_source"] = "gradient_lbfgs"
            evaluated_rows.append(debug_row)
            eval_cache[key] = y_scalar
            print(
                f"[grad_lbfgs] routine {routine_index or '-'} eval {eval_count} "
                f"f={y_scalar:.6f} x={_format_vector(x_full)}"
            )
            return y_scalar

        bounds_active = list(zip(lower[active], upper[active]))
        x0 = base_x[active]

        if maxiter <= 0:
            base_y = float(best_row["S11"])
            if routine_index is not None and routine_total is not None:
                print(
                    f"[grad_lbfgs] skip routine {routine_index}/{routine_total} "
                    f"because maxiter={maxiter} (returning current best point without new evaluations)"
                )
            res_fun = base_y
            res_x = x0.copy()
            nit = 0
            nfev = 0
        else:
            options = {"maxiter": maxiter}
            if maxfun is not None:
                options["maxfun"] = int(maxfun)
            res = minimize(
                objective_active,
                x0=x0,
                method="L-BFGS-B",
                bounds=bounds_active,
                options=options,
            )
            res_fun = float(res.fun)
            res_x = np.asarray(res.x, dtype=float)
            nit = int(getattr(res, "nit", 0))
            nfev = int(getattr(res, "nfev", eval_count))

        x_new = base_x.copy()
        x_new[active] = res_x
        x_new = np.clip(x_new, lower, upper)
        x_new = _round_vector(x_new, decimals=2)

        if active_indices is not None and fixed_point is not None:
            inactive = [i for i in range(dims) if i not in active]
            x_new[inactive] = _round_vector(np.asarray(fixed_point, dtype=float), decimals=2)[inactive]

        if routine_index is not None and routine_total is not None:
            print(
                f"[grad_lbfgs] end routine {routine_index}/{routine_total} "
                f"(remaining: {routine_total - routine_index}, nit: {nit}, nfev: {nfev})"
            )
            print(
                f"[grad_lbfgs] best x={_format_vector(x_new)} f={res_fun:.6f}"
            )

        return x_new, {
            "method": "gradient",
            "solver": "L-BFGS-B",
            "base_y": res_fun,
            "evaluated_rows": evaluated_rows,
            "nit": nit,
            "nfev": nfev,
        }
