from lib_config import AppConfig
import numpy as np
from scipy.optimize import minimize
from typing import Optional, List, Tuple


class EvaluationBudgetExceeded(RuntimeError):
    pass


def _round_vector(values: np.ndarray, decimals: int = 10) -> np.ndarray:
    return np.round(np.asarray(values, dtype=float), decimals=decimals)


def _format_vector(values: np.ndarray, decimals: int) -> str:
    return np.array2string(_round_vector(values, decimals=decimals), precision=decimals, separator=", ")


class GradientSearch:
    def __init__(self, config: AppConfig):
        self.cfg = config
        runtime = getattr(config, "runtime", None)
        self.round_decimals = getattr(runtime, "round_decimals", 10)
        self.fd_rel_step = float(getattr(runtime, "grad_fd_rel_step", 0.05))
        self.explore_step_ratios = tuple(getattr(runtime, "grad_explore_step_ratios", (0.5, 0.25, 0.1)))
        self.lbfgs_maxls = int(getattr(runtime, "grad_lbfgs_maxls", 40))

    def _build_base_point(self, best_x, active_indices, fixed_point):
        base_x = np.asarray(best_x, dtype=float).copy()
        if active_indices is None or fixed_point is None:
            return base_x
        x = _round_vector(np.asarray(fixed_point, dtype=float).copy(), decimals=self.round_decimals)
        active = list(active_indices)
        x[active] = base_x[active]
        return x

    def _estimate_gradient(self, objective_active, x0, bounds_active):
        gradient = np.zeros_like(x0, dtype=float)
        active_span = np.array([max(high - low, 1e-12) for low, high in bounds_active], dtype=float)

        for idx, (low, high) in enumerate(bounds_active):
            delta = min(self.fd_rel_step * active_span[idx], max(high - low, 1e-12))
            if delta <= 0:
                continue

            left = np.asarray(x0, dtype=float).copy()
            right = np.asarray(x0, dtype=float).copy()
            left[idx] = max(low, left[idx] - delta)
            right[idx] = min(high, right[idx] + delta)

            if np.isclose(left[idx], right[idx]):
                continue

            f_left = objective_active(left)
            f_right = objective_active(right)
            gradient[idx] = (f_right - f_left) / (right[idx] - left[idx])

        return gradient, active_span

    def _select_exploratory_start(self, objective_active, x0, bounds_active):
        gradient, active_span = self._estimate_gradient(objective_active, x0, bounds_active)
        grad_norm = np.linalg.norm(gradient)
        best_z = np.asarray(x0, dtype=float).copy()
        best_value = objective_active(best_z)

        if not np.isfinite(grad_norm) or grad_norm == 0:
            return best_z, best_value, gradient, None

        direction = -gradient / grad_norm
        best_ratio = None

        for step_ratio in self.explore_step_ratios:
            candidate = np.asarray(x0, dtype=float) + step_ratio * active_span * direction
            for idx, (low, high) in enumerate(bounds_active):
                candidate[idx] = min(high, max(low, candidate[idx]))
            candidate = _round_vector(candidate, decimals=self.round_decimals)
            candidate_value = objective_active(candidate)
            if candidate_value < best_value:
                best_z = candidate
                best_value = candidate_value
                best_ratio = float(step_ratio)

        return best_z, best_value, gradient, best_ratio

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
        max_evals: Optional[int] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, dict]:
        if objective_func is None:
            raise ValueError("GradientSearch requires objective_func.")

        lower = np.asarray(lower_bounds, dtype=float)
        upper = np.asarray(upper_bounds, dtype=float)
        dims = len(lower)
        active = list(range(dims)) if active_indices is None else list(active_indices)

        best_row = dict(start_row) if start_row is not None else min(history_data, key=lambda row: row["S11"])
        best_x = _round_vector(np.asarray([best_row[name] for name in param_names], dtype=float), decimals=self.round_decimals)
        base_x = self._build_base_point(best_x, active_indices, fixed_point)

        if routine_index is not None and routine_total is not None:
            remaining_before = routine_total - routine_index + 1
            remaining_after = routine_total - routine_index
            print(
                f"[grad_lbfgs] start routine {routine_index}/{routine_total} "
                f"(remaining incl. this: {remaining_before}, remaining after: {remaining_after})"
            )

        bounds_active = list(zip(lower[active], upper[active]))
        x0 = _round_vector(base_x[active], decimals=self.round_decimals)

        evaluated_rows = []
        eval_cache = {}
        eval_count = 0
        best_seen_state = {"x_active": x0.copy(), "y": float(best_row["S11"]) }

        def objective_active(z):
            nonlocal eval_count
            x_full = base_x.copy()
            z_array = np.asarray(z, dtype=float)
            x_full[active] = z_array
            x_full = np.clip(x_full, lower, upper)
            x_full = _round_vector(x_full, decimals=self.round_decimals)

            key = tuple(x_full.tolist())
            if key in eval_cache:
                cached_value = eval_cache[key]
                print(
                    f"[grad_lbfgs] routine {routine_index or '-'} cache hit "
                    f"f={cached_value:.6f} x={_format_vector(x_full, self.round_decimals)}"
                )
                return cached_value

            if max_evals is not None and eval_count >= int(max_evals):
                raise EvaluationBudgetExceeded("gradient search exhausted its evaluation budget")

            y_value, row = objective_func(param_names, x_full)
            y_scalar = float(y_value)
            eval_count += 1
            debug_row = dict(row)
            debug_row["debug_eval_idx"] = eval_count
            evaluated_rows.append(debug_row)
            eval_cache[key] = y_scalar

            if y_scalar < best_seen_state["y"]:
                best_seen_state["y"] = y_scalar
                best_seen_state["x_active"] = x_full[active].copy()

            print(
                f"[grad_lbfgs] routine {routine_index or '-'} eval {eval_count} "
                f"f={y_scalar:.6f} x={_format_vector(x_full, self.round_decimals)}"
            )
            return y_scalar

        exploratory_x0 = x0.copy()
        exploratory_fun = float(best_row["S11"])
        exploratory_grad = np.zeros_like(x0)
        exploratory_ratio = None
        lbfgs_eps = np.array([max(self.fd_rel_step * (high - low), 1e-8) for low, high in bounds_active], dtype=float)
        budget_exhausted = False

        try:
            exploratory_x0, exploratory_fun, exploratory_grad, exploratory_ratio = self._select_exploratory_start(
                objective_active,
                x0,
                bounds_active,
            )
        except EvaluationBudgetExceeded:
            budget_exhausted = True
            best_active = np.asarray(best_seen_state["x_active"], dtype=float)
            exploratory_x0 = best_active.copy()
            exploratory_fun = float(best_seen_state["y"])
            exploratory_grad = np.zeros_like(x0)

        if exploratory_ratio is not None:
            print(
                f"[grad_lbfgs] routine {routine_index or '-'} exploratory start "
                f"ratio={exploratory_ratio:.3f} f={exploratory_fun:.6f} "
                f"x={_format_vector(exploratory_x0, self.round_decimals)}"
            )

        if maxiter <= 0 or budget_exhausted:
            res_fun = exploratory_fun
            res_x = exploratory_x0.copy()
            nit = 0
            nfev = eval_count
        else:
            options = {
                "maxiter": maxiter,
                "eps": lbfgs_eps,
                "maxls": self.lbfgs_maxls,
            }
            if maxfun is not None:
                options["maxfun"] = int(maxfun)
            try:
                res = minimize(
                    objective_active,
                    x0=exploratory_x0,
                    method="L-BFGS-B",
                    bounds=bounds_active,
                    options=options,
                )
                res_fun = float(res.fun)
                res_x = np.asarray(res.x, dtype=float)
                nit = int(getattr(res, "nit", 0))
                nfev = int(getattr(res, "nfev", eval_count))
            except EvaluationBudgetExceeded:
                best_active = np.asarray(best_seen_state["x_active"], dtype=float)
                res_x = best_active.copy()
                res_fun = float(best_seen_state["y"])
                nit = 0
                nfev = eval_count
                budget_exhausted = True

        x_new = base_x.copy()
        x_new[active] = res_x
        x_new = np.clip(x_new, lower, upper)
        x_new = _round_vector(x_new, decimals=self.round_decimals)

        if active_indices is not None and fixed_point is not None:
            inactive = [i for i in range(dims) if i not in active]
            x_new[inactive] = _round_vector(np.asarray(fixed_point, dtype=float), decimals=self.round_decimals)[inactive]

        if routine_index is not None and routine_total is not None:
            print(
                f"[grad_lbfgs] end routine {routine_index}/{routine_total} "
                f"(remaining: {routine_total - routine_index}, nit: {nit}, nfev: {nfev}, budget_exhausted: {budget_exhausted})"
            )
            print(
                f"[grad_lbfgs] best x={_format_vector(x_new, self.round_decimals)} f={res_fun:.6f}"
            )

        return x_new, {
            "method": "gradient",
            "solver": "L-BFGS-B",
            "base_y": res_fun,
            "evaluated_rows": evaluated_rows,
            "nit": nit,
            "nfev": nfev,
            "exploratory_start_ratio": exploratory_ratio,
            "exploratory_grad_norm": float(np.linalg.norm(exploratory_grad)),
            "budget_exhausted": budget_exhausted,
        }
