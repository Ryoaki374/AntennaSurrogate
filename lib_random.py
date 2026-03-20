from lib_config import AppConfig
import numpy as np
from typing import Optional, List, Tuple


class RandomSearch:
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
        **kwargs,
    ) -> Tuple[np.ndarray, dict]:
        lower = np.asarray(lower_bounds, dtype=float)
        upper = np.asarray(upper_bounds, dtype=float)
        dims = len(lower)

        if active_indices is None or len(active_indices) == dims:
            x_new = np.random.uniform(lower, upper)
        else:
            if fixed_point is None:
                raise ValueError("fixed_point is required when active_indices is provided.")
            x_new = np.asarray(fixed_point, dtype=float).copy()
            active = list(active_indices)
            x_new[active] = np.random.uniform(lower[active], upper[active])

        return x_new, {"method": "random", "evaluated_rows": []}
