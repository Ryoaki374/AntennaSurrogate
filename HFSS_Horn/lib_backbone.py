# Import AppConfig from lib_config.py
from lib_config import AppConfig
from datetime import datetime
from scipy.stats.qmc import LatinHypercube, scale
import h5py
import os
import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Sequence, Mapping

import lib_RFdesign


SECTION_FRAC_SLICE = slice(2, 7)
SECTION_FRAC_NAMES = ("f_wg", "f_t1", "f_mid", "f_t2", "f_ap")
SECTION_FRAC_ATOL = 1.0e-8
FIXED_D_APERTURE = 11.6
FIXED_D_WAVEGUIDE = 1.8
DIAMETER_ATOL = 1.0e-6
MIN_SECTION_FRAC = 1.0e-6


def section_frac_sum_constraint(x):
    """
    Equality constraint for horn section fractions:
        f_wg + f_t1 + f_mid + f_t2 + f_ap = 1.0
    """
    f_wg, f_t1, f_mid, f_t2, f_ap = np.asarray(x, dtype=float).flatten()[SECTION_FRAC_SLICE]
    return f_wg + f_t1 + f_mid + f_t2 + f_ap - 1.0


def _project_to_bounded_simplex(values, lower, upper, target=1.0, tol=1.0e-12, max_iter=100):
    """Project values onto {x | sum(x)=target, lower<=x<=upper}."""
    values = np.asarray(values, dtype=float).reshape(-1)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)
    if values.shape != lower.shape or values.shape != upper.shape:
        raise ValueError("values, lower, and upper must have the same shape.")
    if np.any(lower > upper):
        raise ValueError("lower bounds must be <= upper bounds for section fractions.")
    if lower.sum() - target > tol or target - upper.sum() > tol:
        raise ValueError("Section fraction bounds cannot satisfy the sum-to-one constraint.")

    lo = np.min(values - upper)
    hi = np.max(values - lower)
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
            projected = np.clip(projected, lower, upper)
    return projected


def enforce_section_frac_constraint(x, lower_bounds=None, upper_bounds=None, decimals=None):
    """Return a copy of x with horn section fractions projected to sum exactly to 1."""
    arr = np.asarray(x, dtype=float).flatten().copy()
    if arr.size < SECTION_FRAC_SLICE.stop:
        raise ValueError("Horn parameter vector must contain d_m, l_tot, and five section fractions.")

    # d_m is optimized, while d_aperture and d_waveguide are fixed. Keep
    # generated/suggested values inside the strict geometry requirement used by
    # ConvexHorn: d_waveguide < d_m < d_aperture.
    arr[0] = np.clip(
        arr[0],
        FIXED_D_WAVEGUIDE + DIAMETER_ATOL,
        FIXED_D_APERTURE - DIAMETER_ATOL,
    )

    if lower_bounds is None:
        lower = np.full(5, MIN_SECTION_FRAC, dtype=float)
    else:
        lower = np.maximum(
            np.asarray(lower_bounds, dtype=float).flatten()[SECTION_FRAC_SLICE],
            MIN_SECTION_FRAC,
        )
    if upper_bounds is None:
        upper = np.ones(5, dtype=float)
    else:
        upper = np.asarray(upper_bounds, dtype=float).flatten()[SECTION_FRAC_SLICE]

    arr[SECTION_FRAC_SLICE] = _project_to_bounded_simplex(
        arr[SECTION_FRAC_SLICE], lower=lower, upper=upper, target=1.0
    )
    if decimals is not None:
        arr = np.round(arr, int(decimals))
        arr[0] = np.clip(
            arr[0],
            FIXED_D_WAVEGUIDE + DIAMETER_ATOL,
            FIXED_D_APERTURE - DIAMETER_ATOL,
        )
        # Rounding can introduce a tiny sum error. Correct the largest free fraction.
        residual = 1.0 - arr[SECTION_FRAC_SLICE].sum()
        if abs(residual) > 0.0:
            frac = arr[SECTION_FRAC_SLICE].copy()
            lower = np.round(lower, int(decimals))
            upper = np.round(upper, int(decimals))
            candidates = np.where((frac + residual >= lower) & (frac + residual <= upper))[0]
            idx = int(candidates[np.argmax(frac[candidates])]) if candidates.size else int(np.argmax(frac))
            frac[idx] = np.round(frac[idx] + residual, int(decimals))
            arr[SECTION_FRAC_SLICE] = frac
    return arr


class Backbone:

    def __init__(self, config: AppConfig,):
        self.cfg = config
        self.h5f = None
        self.current_sim_id = 0

    def _round_param_values(self, param_values: Sequence[float], decimals: int = 10) -> np.ndarray:
        return np.round(np.asarray(param_values, dtype=float).flatten(), decimals=decimals)

    def _enforce_horn_constraints(self, param_values: Sequence[float], decimals: Optional[int] = None) -> np.ndarray:
        return enforce_section_frac_constraint(
            param_values,
            lower_bounds=self.cfg.hfss.lower_bounds,
            upper_bounds=self.cfg.hfss.upper_bounds,
            decimals=decimals,
        )

    def mkdir(self):
        if not hasattr(self, "dir_run"):
            timestamp = datetime.now().strftime("%m%d%H%M%S")
            self.dir_run = self.cfg.env.dir_base / f"{timestamp}"
            self.dir_run.mkdir(exist_ok=True)
            print(f"Created new run directory: {self.dir_run}")

    def _get_dir_run(self):
        return self.dir_run

    def _get_path_models(self):
        base = self.dir_run
        files = self.cfg.hfss.filename_models
        return [base / f for f in files], [str(base / f) for f in files]

    def initStorer(self, runs_dir = None, mode = "w"):
        """Initializes settings for saving data to an HDF5 file."""
        self.mkdir()
        if runs_dir is None:
            runs_dir = self.dir_run
        filepath = runs_dir / "results.h5"
        self.h5f = h5py.File(filepath, mode)
        self.h5f.create_group("input")
        self.h5f.create_group("output")
        self.h5f.create_group("learning_curve")
        print(f"HDF5 dataset created at: {filepath}")

    def _addNewDatasetToHDF(self, df: pd.DataFrame, grp_name_str: str, dset_name_str: str):
        grp = self.h5f[grp_name_str]
        if dset_name_str in grp:
            return grp[dset_name_str]
        data = df.to_numpy(dtype=np.float32)
        n_rows, n_cols = data.shape
        dset = grp.create_dataset(
            dset_name_str,
            shape=(n_rows, n_cols),
            dtype=np.float32,
            compression="gzip",
        )
        dset.attrs["columns"] = json.dumps(df.columns.tolist())
        dset[:, :] = data
        print(f"wrote {n_rows} rows x {n_cols} cols to {dset_name_str}")

    def _getSimulationID(self):
        self.current_sim_id += 1
        return self.current_sim_id

    def call_subroutine(self, config, index, param_names, param_values, value_fmt=None):
        model_paths, _ = self._get_path_models()
        temp_file = str(self.dir_run / self.cfg.io.filename_temp)

        if len(model_paths) != 1:
            raise ValueError(f"Horn workflow expects exactly one model path, got {len(model_paths)}: {model_paths}")

        group_order = self.cfg.hfss.group_order or list(self.cfg.hfss.param_groups.keys())
        grouped_values = {}
        round_decimals = getattr(getattr(self.cfg, "runtime", None), "round_decimals", 10)
        if value_fmt is None:
            value_fmt = f"{{:.{round_decimals}f}}"
        param_values = self._round_param_values(param_values, decimals=round_decimals)
        param_values = self._enforce_horn_constraints(param_values, decimals=round_decimals)

        cursor = 0
        for group_name in group_order:
            group_cfg = self.cfg.hfss.param_groups[group_name]
            names = list(group_cfg["param_names"])
            next_cursor = cursor + len(names)
            values = param_values[cursor:next_cursor]
            if len(values) != len(names):
                raise ValueError(
                    f"Group {group_name} expects {len(names)} parameters, got {len(values)} from index {cursor}."
                )
            grouped_values[group_name] = dict(zip(names, values))
            cursor = next_cursor

        if cursor != len(param_values):
            raise ValueError(f"Expected {cursor} HFSS parameters based on param_groups, got {len(param_values)}.")

        horn_group = grouped_values["Horn"]
        section_fracs = tuple(float(horn_group[name]) for name in SECTION_FRAC_NAMES)
        constraint_residual = section_frac_sum_constraint(param_values)
        if abs(constraint_residual) > SECTION_FRAC_ATOL:
            raise ValueError(f"section_fracs must sum to 1.0; residual={constraint_residual}.")

        design = lib_RFdesign.ConvexHorn(model_path=model_paths[0])
        d_middle = float(horn_group.get("d_m", horn_group.get("d_middle")))
        total_length = float(horn_group.get("l_tot", horn_group.get("total_length")))

        design.genHorn(
            d_aperture=FIXED_D_APERTURE,
            d_middle=d_middle,
            d_waveguide=FIXED_D_WAVEGUIDE,
            total_length=total_length,
            section_fracs=section_fracs,
        )

        while True:
            if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                time.sleep(0.5)
                print("  > Result received from HFSS.")
                return True
            time.sleep(1)

    def LHSsampler(self, dims, nums, lower_bounds, upper_bounds):
        sampler = LatinHypercube(d=dims,)
        samples_continuous = sampler.random(n=nums)
        X_initial = scale(samples_continuous, lower_bounds, upper_bounds)
        if dims >= SECTION_FRAC_SLICE.stop:
            X_initial = np.vstack([
                enforce_section_frac_constraint(x, lower_bounds, upper_bounds) for x in X_initial
            ])
        return X_initial

    def LHSsampler_extended(self, dims: int, nums: int, lower_bounds, upper_bounds, active_indices: Optional[List[int]] = None, fixed_point: Optional[np.ndarray] = None, fixed_points: Optional[np.ndarray] = None) -> np.ndarray:
        X_fixed = self._as2dPoints(fixed_points, dims)
        n_fixed = 0 if X_fixed is None else min(len(X_fixed), nums)
        rows = []
        if n_fixed:
            rows.append(X_fixed[:n_fixed])
        n_needed = nums - n_fixed
        if n_needed <= 0:
            return np.vstack(rows) if rows else np.empty((0, dims))

        if active_indices is None or len(active_indices) == dims:
            X_gen = self.LHSsampler(dims, n_needed, lower_bounds, upper_bounds)
        else:
            if fixed_point is None:
                raise ValueError("fixed_point is required when active_indices is provided.")
            active = list(active_indices)
            free_dims = len(active)
            lb, ub = self._sliceBounds(lower_bounds, upper_bounds, active)
            X_free = self.LHSsampler(free_dims, n_needed, lb, ub)
            X_gen = self._tileFixedPoint(fixed_point, n_needed)
            X_gen[:, active] = X_free

        if dims >= SECTION_FRAC_SLICE.stop:
            X_gen = np.vstack([
                enforce_section_frac_constraint(x, lower_bounds, upper_bounds) for x in X_gen
            ])
        rows.append(X_gen)
        return np.vstack(rows)

    def _buildSamplingIndices(self, dims: int, param_groups: Mapping[str, Mapping[str, Any]], group_order: Optional[List[str]] = None,) -> Tuple[List[int], np.ndarray, Dict[str, List[int]]]:
        order = group_order or list(param_groups.keys())
        fixed_point = np.empty((dims,), dtype=float)
        group_indices: Dict[str, List[int]] = {}
        cursor = 0
        active_set = set()
        for gname in order:
            g = param_groups[gname]
            names = list(g["param_names"])
            baseline = list(g["baseline"])
            vary = bool(g["vary"])
            m = len(names)
            idxs = list(range(cursor, cursor + m))
            group_indices[gname] = idxs
            for i, v in zip(idxs, baseline):
                fixed_point[i] = float(v)
            if vary:
                active_set.update(idxs)
            cursor += m
        if dims >= SECTION_FRAC_SLICE.stop:
            fixed_point = enforce_section_frac_constraint(
                fixed_point, self.cfg.hfss.lower_bounds, self.cfg.hfss.upper_bounds
            )
        active_indices = sorted(active_set)
        return active_indices, fixed_point, group_indices

    def _as2dPoints(self, points, dims: int) -> Optional[np.ndarray]:
        if points is None:
            return None
        arr = np.asarray(points, dtype=float)
        if arr.size == 0:
            return None
        arr = arr.reshape(1, -1) if arr.ndim == 1 else arr
        if arr.shape[1] != dims:
            raise ValueError(f"fixed_points must have {dims} columns, got {arr.shape[1]}.")
        return arr

    def _tileFixedPoint(self, fixed_point: np.ndarray, n: int) -> np.ndarray:
        return np.tile(np.asarray(fixed_point, dtype=float), (n, 1))

    def _sliceBounds(self, lower_bounds, upper_bounds, active_indices: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        lb = np.asarray(lower_bounds, dtype=float)[list(active_indices)]
        ub = np.asarray(upper_bounds, dtype=float)[list(active_indices)]
        return lb, ub

    def _genOutputDataFrame(self, df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df, pd.DataFrame):
            return df.copy()
        return pd.DataFrame(df)

    def in_bounds(self, x, lb, ub):
        x = np.asarray(x, dtype=float)
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)
        return np.all((x >= lb) & (x <= ub))

    def all_in_bounds(self, xs, lb, ub):
        return all(self.in_bounds(x, lb, ub) for x in xs)

    def printn(self, text: str):
        print("\n" + "=" * 75)
        print(text)
        print("=" * 75)
