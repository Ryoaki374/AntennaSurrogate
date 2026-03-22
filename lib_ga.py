from lib_config import AppConfig
import numpy as np
from typing import Optional, List, Tuple
from deap import base, creator, tools, benchmarks, cma

if not hasattr(creator, "AntennaFitnessMin"):
    creator.create("AntennaFitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "AntennaIndividual"):
    creator.create("AntennaIndividual", list, fitness=creator.AntennaFitnessMin)


class RealCodedGA:
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
        sigma: Optional[float] = None,
        lambda_: Optional[int] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, dict]:
        if objective_func is None:
            raise ValueError("RealCodedGA requires objective_func for CMA-ES search.")

        lower = np.asarray(lower_bounds, dtype=float)
        upper = np.asarray(upper_bounds, dtype=float)
        dims = len(lower)
        active = list(range(dims)) if active_indices is None else list(active_indices)

        best_row = min(history_data, key=lambda row: row["S11"])
        best_x = np.asarray([best_row[name] for name in param_names], dtype=float)
        base_x = self._build_base_point(best_x, active_indices, fixed_point)

        active_centroid = base_x[active]
        active_span = upper[active] - lower[active]
        sigma = float(sigma if sigma is not None else max(np.mean(active_span) * 0.2, 1e-3))
        lambda_ = int(lambda_ if lambda_ is not None else max(4, 4 + int(3 * np.log(len(active) + 1))))

        toolbox = base.Toolbox()
        strategy = cma.Strategy(centroid=active_centroid.tolist(), sigma=sigma, lambda_=lambda_)
        toolbox.register("generate", strategy.generate, creator.AntennaIndividual)
        toolbox.register("update", strategy.update)

        evaluated_rows = []
        eval_cache = {}

        def _cache_key(x_full):
            return tuple(np.round(np.asarray(x_full, dtype=float), 12))

        population = toolbox.generate()
        for individual in population:
            z = np.asarray(individual, dtype=float)
            x_full = base_x.copy()
            x_full[active] = z
            x_full = np.clip(x_full, lower, upper)
            key = _cache_key(x_full)
            if key in eval_cache:
                y_scalar = eval_cache[key]["y"]
            else:
                y_value, row = objective_func(param_names, x_full)
                y_scalar = float(y_value)
                eval_cache[key] = {"y": y_scalar, "row": row.copy()}
                evaluated_rows.append(row.copy())
            individual[:] = x_full[active].tolist()
            individual.fitness.values = (y_scalar,)

        toolbox.update(population)
        best_individual = tools.selBest(population, 1)[0]

        x_new = base_x.copy()
        x_new[active] = np.asarray(best_individual, dtype=float)
        x_new = np.clip(x_new, lower, upper)
        if active_indices is not None and fixed_point is not None:
            inactive = [i for i in range(dims) if i not in active]
            x_new[inactive] = np.asarray(fixed_point, dtype=float)[inactive]

        final_key = _cache_key(x_new)
        final_eval = eval_cache.get(final_key)

        return x_new, {
            "method": "cmaes",
            "solver": "deap.cma",
            "sigma": sigma,
            "lambda": lambda_,
            "base_y": float(best_individual.fitness.values[0]),
            "evaluated_rows": evaluated_rows,
            "final_row": None if final_eval is None else final_eval["row"].copy(),
            "final_y": None if final_eval is None else float(final_eval["y"]),
        }
