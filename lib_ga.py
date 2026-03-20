from lib_config import AppConfig
import numpy as np
from typing import Optional, List, Tuple


class RealCodedGA:
    def __init__(self, config: AppConfig):
        self.cfg = config

    def search(
        self,
        history_data,
        param_names,
        lower_bounds,
        upper_bounds,
        objective_func=None,
        active_indices: Optional[List[int]] = None,
        fixed_point: Optional[np.ndarray] = None,
        elite_size: int = 4,
        mutation_scale: float = 0.05,
        **kwargs,
    ) -> Tuple[np.ndarray, dict]:
        lower = np.asarray(lower_bounds, dtype=float)
        upper = np.asarray(upper_bounds, dtype=float)
        dims = len(lower)
        rows = sorted(history_data, key=lambda row: row["S11"])
        elites = rows[: max(2, min(elite_size, len(rows)))]

        p1 = np.asarray([elites[np.random.randint(len(elites))][name] for name in param_names], dtype=float)
        p2 = np.asarray([elites[np.random.randint(len(elites))][name] for name in param_names], dtype=float)
        alpha = np.random.uniform(0.0, 1.0, size=dims)
        child = alpha * p1 + (1.0 - alpha) * p2
        child += np.random.normal(0.0, mutation_scale * (upper - lower), size=dims)
        child = np.clip(child, lower, upper)

        if active_indices is not None and len(active_indices) != dims:
            if fixed_point is None:
                raise ValueError("fixed_point is required when active_indices is provided.")
            x_new = np.asarray(fixed_point, dtype=float).copy()
            active = list(active_indices)
            x_new[active] = child[active]
        else:
            x_new = child

        return x_new, {"method": "ga", "evaluated_rows": [], "elite_size": len(elites)}
