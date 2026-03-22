from lib_config import AppConfig
import numpy as np
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
        fd_eps: float = 0.02,
        step_scale: float = 0.05,
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

        best_row = min(history_data, key=lambda row: row["S11"])
        best_x = np.asarray([best_row[name] for name in param_names], dtype=float)
        base_x = self._build_base_point(best_x, active_indices, fixed_point)
        base_y = float(best_row["S11"])

        if routine_index is not None and routine_total is not None:
            remaining_before = routine_total - routine_index + 1
            remaining_after = routine_total - routine_index
            print(
                f"[grad_db] start routine {routine_index}/{routine_total} "
                f"(remaining incl. this: {remaining_before}, remaining after: {remaining_after})"
            )

        grad = np.zeros(dims, dtype=float)
        evaluated_rows = []

        for idx in active:
            delta = fd_eps * max(upper[idx] - lower[idx], 1e-12)
            x_probe = base_x.copy()
            x_probe[idx] = min(upper[idx], x_probe[idx] + delta)
            if x_probe[idx] == base_x[idx]:
                x_probe[idx] = max(lower[idx], x_probe[idx] - delta)
            denom = x_probe[idx] - base_x[idx]
            if denom == 0:
                continue
            y_probe, row_probe = objective_func(param_names, x_probe)
            grad[idx] = (float(y_probe) - base_y) / denom
            evaluated_rows.append(row_probe)

        grad_active = grad[active]
        grad_norm = np.linalg.norm(grad_active)
        x_new = base_x.copy()
        if grad_norm > 0:
            step = step_scale * (upper[active] - lower[active]) * (grad_active / grad_norm)
            x_new[active] = x_new[active] - step
        x_new = np.clip(x_new, lower, upper)

        if active_indices is not None and fixed_point is not None:
            inactive = [i for i in range(dims) if i not in active]
            x_new[inactive] = np.asarray(fixed_point, dtype=float)[inactive]

        if routine_index is not None and routine_total is not None:
            print(
                f"[grad_db] end routine {routine_index}/{routine_total} "
                f"(remaining: {routine_total - routine_index})"
            )

        return x_new, {
            "method": "gradient",
            "gradient": grad.tolist(),
            "base_y": base_y,
            "evaluated_rows": evaluated_rows,
        }
