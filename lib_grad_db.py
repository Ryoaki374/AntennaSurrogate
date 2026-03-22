from lib_config import AppConfig
import numpy as np
from typing import Optional, List, Tuple


class EvaluationBudgetExceeded(RuntimeError):
    pass


def _round_vector(values: np.ndarray, decimals: int = 10) -> np.ndarray:
    return np.round(np.asarray(values, dtype=float), decimals=decimals)


class GradientSearch:
    def __init__(self, config: AppConfig):
        self.cfg = config
        runtime = getattr(config, "runtime", None)
        self.round_decimals = getattr(runtime, "round_decimals", 10)
        self.fd_rel_step = float(getattr(runtime, "grad_db_fd_rel_step", 0.02))
        self.initial_step_ratio = float(getattr(runtime, "grad_db_init_step_ratio", 0.25))
        self.backtrack_beta = float(getattr(runtime, "grad_db_backtrack_beta", 0.5))
        self.armijo_c = float(getattr(runtime, "grad_db_armijo_c", 1e-4))
        self.min_step_ratio = float(getattr(runtime, "grad_db_min_step_ratio", 1e-3))

    def _build_base_point(self, best_x, active_indices, fixed_point):
        base_x = np.asarray(best_x, dtype=float).copy()
        if active_indices is None or fixed_point is None:
            return base_x
        x = _round_vector(np.asarray(fixed_point, dtype=float).copy(), decimals=self.round_decimals)
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
        fd_eps: Optional[float] = None,
        step_scale: Optional[float] = None,
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
                f"[grad_db] start routine {routine_index}/{routine_total} "
                f"(remaining incl. this: {remaining_before}, remaining after: {remaining_after})"
            )

        evaluated_rows = []
        eval_cache = {}
        eval_count = 0

        def evaluate(x_full):
            nonlocal eval_count
            x_eval = _round_vector(np.clip(np.asarray(x_full, dtype=float), lower, upper), decimals=self.round_decimals)
            key = tuple(x_eval.tolist())
            if key in eval_cache:
                return eval_cache[key]
            if max_evals is not None and eval_count >= int(max_evals):
                raise EvaluationBudgetExceeded("gradient_db exhausted its evaluation budget")
            y_value, row = objective_func(param_names, x_eval)
            y_scalar = float(y_value)
            eval_count += 1
            debug_row = dict(row)
            debug_row["debug_eval_idx"] = eval_count
            evaluated_rows.append(debug_row)
            eval_cache[key] = (y_scalar, debug_row)
            print(
                f"[grad_db] routine {routine_index or '-'} eval {eval_count} "
                f"f={y_scalar:.6f} x={x_eval.tolist()}"
            )
            return eval_cache[key]

        budget_exhausted = False
        try:
            base_y, _ = evaluate(base_x)
            grad = np.zeros(dims, dtype=float)
            active_span = np.maximum(upper[active] - lower[active], 1e-12)
            fd_rel_step = float(fd_eps) if fd_eps is not None else self.fd_rel_step
            initial_step_ratio = float(step_scale) if step_scale is not None else self.initial_step_ratio

            for local_idx, idx in enumerate(active):
                delta = fd_rel_step * active_span[local_idx]
                x_left = base_x.copy()
                x_right = base_x.copy()
                x_left[idx] = max(lower[idx], x_left[idx] - delta)
                x_right[idx] = min(upper[idx], x_right[idx] + delta)
                denom = x_right[idx] - x_left[idx]
                if denom <= 0:
                    continue
                y_left, _ = evaluate(x_left)
                y_right, _ = evaluate(x_right)
                grad[idx] = (y_right - y_left) / denom

            grad_active = grad[active]
            grad_norm = np.linalg.norm(grad_active)
            x_new = base_x.copy()
            best_step_ratio = 0.0

            if grad_norm > 0:
                direction = -grad_active / grad_norm
                directional_derivative = float(np.dot(grad_active, direction))
                step_ratio = initial_step_ratio

                while step_ratio >= self.min_step_ratio:
                    candidate = base_x.copy()
                    candidate[active] = candidate[active] + step_ratio * active_span * direction
                    candidate = _round_vector(np.clip(candidate, lower, upper), decimals=self.round_decimals)
                    candidate_y, _ = evaluate(candidate)
                    armijo_rhs = base_y + self.armijo_c * step_ratio * directional_derivative * np.linalg.norm(active_span)
                    if candidate_y <= armijo_rhs:
                        x_new = candidate
                        best_step_ratio = step_ratio
                        break
                    step_ratio *= self.backtrack_beta
        except EvaluationBudgetExceeded:
            grad = np.zeros(dims, dtype=float)
            grad_norm = 0.0
            x_new = base_x.copy()
            best_step_ratio = 0.0
            budget_exhausted = True

        if active_indices is not None and fixed_point is not None:
            inactive = [i for i in range(dims) if i not in active]
            x_new[inactive] = _round_vector(np.asarray(fixed_point, dtype=float), decimals=self.round_decimals)[inactive]

        final_y, _ = evaluate(x_new)

        if routine_index is not None and routine_total is not None:
            print(
                f"[grad_db] end routine {routine_index}/{routine_total} "
                f"(remaining: {routine_total - routine_index}, step_ratio: {best_step_ratio:.6f}, nfev: {eval_count})"
            )

        return x_new, {
            "method": "gradient_db",
            "gradient": grad.tolist(),
            "base_y": final_y,
            "evaluated_rows": evaluated_rows,
            "step_ratio": best_step_ratio,
            "grad_norm": float(grad_norm),
            "nfev": eval_count,
            "budget_exhausted": budget_exhausted,
        }
