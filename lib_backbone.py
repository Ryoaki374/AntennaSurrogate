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

class Backbone:

    def __init__(self, config: AppConfig,):
        self.cfg = config
        self.h5f = None
        self.current_sim_id = 0

    def mkdir(self):
        if not hasattr(self, "dir_run"): # attribute check
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
        grp_input = self.h5f.create_group("input")
        grp_output = self.h5f.create_group("output")
        grp_lc = self.h5f.create_group("learning_curve")
        print(f"HDF5 dataset created at: {filepath}")

    def _addNewDatasetToHDF(self, df: pd.DataFrame, grp_name_str: str, dset_name_str: str):
        
        grp = self.h5f[grp_name_str]

        if dset_name_str in grp:
            return grp[dset_name_str]
        
        data = df.to_numpy(dtype=np.float32) # without header
        n_rows, n_cols = data.shape

        dset = grp.create_dataset(
            dset_name_str,
            shape=(n_rows, n_cols),
            dtype=np.float32,
            compression="gzip",
        )

        # column name
        dset.attrs["columns"] = json.dumps(df.columns.tolist())

        # write data
        dset[:, :] = data

        print(f"wrote {n_rows} rows x {n_cols} cols to {dset_name_str}")

    def _getSimulationID(self):
        self.current_sim_id += 1
        return self.current_sim_id

    '''
    def call_subroutine(self, config, index, param_names, param_values, value_fmt="{:.2f}"):
        modelpaths = self._get_path_models()
        #input_file = config["INPUT_FILE"]
        input_file = str(self.dir_run / self.cfg.io.filename_input)
        #results_file = config["RESULTS_FILE"]
        #temp_file = config["TEMP_FILE"]
        temp_file = str(self.dir_run / self.cfg.io.filename_temp)
        #unit_arr = config["param_units"]
        unit_arr = self.cfg.hfss.param_units

        #param_names = self.cfg.hfss.param_names
        param_names_step = self.cfg.hfss.param_names[-2:]
        param_values_step = param_values[-2:]
        
        param_names = self.cfg.hfss.param_names[:4]
        param_values = param_values[:4]

        unit_arr = unit_arr[:4]

        # Create step file for Backshort
        design = lib_RFdesign.ConvexBackshort(model_path=modelpaths[0])
        a = 9.525
        b = 4.7625
        c = param_values_step[0]
        k = int(param_values_step[1])
        convex_backshort = design.genBackshort(a=a, b=b, c=c, k=k, grid_res=30, shifts=(0, -4.7625, -0.34575))
        #design.plotConvex3D(convex_backshort) 
        # 
        # Create step file for Finshape
        design = lib_RFdesign.ConvexFinshape(model_path=modelpaths[1])
        a = param_values[0]
        b = param_values[1]
        k = param_values[2]
        convex_finshape = design.genFinshape(a=a, b=b, k=k, grid_res=400, shifts=(0.0, -1.0))
        design.plotProfile2D(convex_finshape)

        row = {'*': index}
        
        for k, v, u in zip(self.cfg.hfss.param_names, param_values, unit_arr):
            formatted_val = value_fmt.format(float(v))
            row[k] = f"{formatted_val}{u}" if u else formatted_val
    
        pd.DataFrame([row]).to_csv(input_file, index=False)
    
        while True:
            if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                time.sleep(0.5) 
                print("  > Result received from HFSS.")
                return True
            time.sleep(1)
    '''

    def call_subroutine(self, config, index, param_names, param_values, value_fmt="{:.2f}"):
        model_paths, _ = self._get_path_models()
        temp_file = str(self.dir_run / self.cfg.io.filename_temp)

        group_order = self.cfg.hfss.group_order or list(self.cfg.hfss.param_groups.keys())
        grouped_values = {}
        param_values = np.asarray(param_values, dtype=float).flatten()

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

        # Create step file for Backshort
        design = lib_RFdesign.ConvexBackshort(model_path=model_paths[0])
        backshort_group = grouped_values["A"]
        step_param_names = self.cfg.hfss.param_groups["A"]["param_names"]
        step_heights = tuple(float(backshort_group[name]) for name in step_param_names)
        design.genStepBackshort(
            a=9.525,
            b=4.7625,
            step_heights=step_heights,
            shrink=1.5,
            shifts=(0, -4.7625, -0.34575),
        )

        # Create step file for Finshape
        design = lib_RFdesign.ConvexFinshape(model_path=model_paths[1])
        finshape_group = grouped_values["B"]
        convex_finshape = design.genFinshape(
            a=float(finshape_group["a"]),
            b=float(finshape_group["b"]),
            k=float(finshape_group["k"]),
            grid_res=400,
            shifts=(0.0, -1.0),
        )
        #design.plotProfile2D(convex_finshape)

        #row = {'*': index}
        #
        #for k, v, u in zip(self.cfg.hfss.param_names, param_values, unit_arr):
        #    formatted_val = value_fmt.format(float(v))
        #    row[k] = f"{formatted_val}{u}" if u else formatted_val
    
        #pd.DataFrame([row]).to_csv(input_file, index=False) should be detelted
    
        while True:
            if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                time.sleep(0.5) 
                print("  > Result received from HFSS.")
                return True
            time.sleep(1)

    def LHSsampler(self, dims, nums, lower_bounds, upper_bounds):
        sampler = LatinHypercube(d = dims,)
        samples_continuous = sampler.random(n = nums)
        X_initial = scale(samples_continuous, lower_bounds, upper_bounds)
        return X_initial
    
    def LHSsampler_extended(self, dims: int, nums: int, lower_bounds, upper_bounds, active_indices: Optional[List[int]] = None, fixed_point: Optional[np.ndarray] = None, ) -> np.ndarray:
        """
        Extended LHS sampler:
          - If active_indices is None: sample all dims via base LHSsampler.
          - Else: sample only active dims, and fill inactive dims with fixed_point.
        """
        if active_indices is None or len(active_indices) == dims:
            return self.LHSsampler(dims, nums, lower_bounds, upper_bounds)

        if fixed_point is None:
            raise ValueError("fixed_point is required when active_indices is provided.")

        active = list(active_indices)
        free_dims = len(active)

        # 1) sample only free dims using the base sampler
        lb, ub = self._sliceBounds(lower_bounds, upper_bounds, active)
        X_free = self.LHSsampler(free_dims, nums, lb, ub)  # (nums, free_dims)

        # 2) build full matrix from fixed_point and override active dims
        X = self._tileFixedPoint(fixed_point, nums)          # (nums, dims)
        X[:, active] = X_free
        return X
    
    def LHSsampler_extended(self, dims: int, nums: int, lower_bounds, upper_bounds, active_indices: Optional[List[int]] = None, fixed_point: Optional[np.ndarray] = None, fixed_points: Optional[np.ndarray] = None) -> np.ndarray:
        
        X_fixed = self._as2dPoints(fixed_points, dims)
        n_fixed = len(X_fixed)
        n_needed = max(0, nums - n_fixed)

        # If we already have enough fixed points, just return them
        if n_needed == 0:
            return X_fixed[:nums]

        # full LHS
        if active_indices is None or len(active_indices) == dims:
            X_gen = self.LHSsampler(dims, n_needed, lower_bounds, upper_bounds)
            return self._mergePoints(X_fixed, X_gen, nums)

        # partial LHS with fixed dims
        if fixed_point is None:
            raise ValueError("fixed_point is required when active_indices is provided.")

        active = list(active_indices)
        free_dims = len(active)

        # 1) sample only free dims using the base sampler
        lb, ub = self._sliceBounds(lower_bounds, upper_bounds, active)
        X_free = self.LHSsampler(free_dims, n_needed, lb, ub)  # (n_needed, free_dims)

        # 2) build full matrix from fixed_point and override active dims
        X_gen = self._tileFixedPoint(fixed_point, n_needed)       # (n_needed, dims)
        X_gen[:, active] = X_free

        return self._mergePoints(X_fixed, X_gen, nums)
    
    def _buildSamplingIndices(self, dims: int, param_groups: Mapping[str, Mapping[str, Any]], group_order: Optional[List[str]] = None,) -> Tuple[List[int], np.ndarray, Dict[str, List[int]]]:

        # determine the order of groups to process
        order = group_order or list(param_groups.keys())

        fixed_point = np.empty((dims,), dtype=float)
        group_indices: Dict[str, List[int]] = {}

        cursor = 0 # dummy values for fixed_point; will be overwritten by actual baseline values
        active_set = set() # to collect indices of parameters that noted as "vary" in any group

        for gname in order:

            # read
            g = param_groups[gname]
            names = list(g["param_names"])
            baseline = list(g["baseline"])
            vary = bool(g["vary"])

            # create indices for this group
            m = len(names)
            idxs = list(range(cursor, cursor + m))
            group_indices[gname] = idxs

            # set fixed point values for this group
            for i, v in zip(idxs, baseline):
                fixed_point[i] = float(v)

            # if this group is marked as "vary", add its indices to the active set
            if vary:
                active_set.update(idxs)

            cursor += m

        active_indices = sorted(active_set)
        return active_indices, fixed_point, group_indices
    

    def _as2dPoints(self, x: Optional[np.ndarray], dims: int) -> np.ndarray:
        """
        None -> (0, dims)
        (dims,) -> (1, dims)
        (n, dims) -> (n, dims)
        """
        if x is None:
            return np.empty((0, dims), dtype=float)
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr
    
    def _tileFixedPoint(self, fixed_point: np.ndarray, n: int) -> np.ndarray: # copy fixed_point
        # fixed_point: (dims,)
        return np.tile(np.asarray(fixed_point, dtype=float), (n, 1))
    
    def _sliceBounds(self, lower_bounds, upper_bounds, active_indices: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        lb = np.asarray(lower_bounds, dtype=float)[list(active_indices)]
        ub = np.asarray(upper_bounds, dtype=float)[list(active_indices)]
        return lb, ub
    
    def _mergePoints(self, X_fixed: np.ndarray, X_gen: np.ndarray, nums: int) -> np.ndarray:
        if len(X_fixed) == 0:
            return X_gen[:nums]
        if len(X_gen) == 0:
            return X_fixed[:nums]
        X = np.vstack([X_fixed, X_gen])
        return X[:nums]


    def _genOutputDataFrame(self, df_current: pd.DataFrame,):
        df_output = df_current.copy()
        df_output["best"] = df_output["S11"].cummin()
        return df_output
    
    def in_bounds(self, x, lb, ub):
        return all(l <= v <= u for v, l, u in zip(x, lb, ub))

    def all_in_bounds(self, xs, lb, ub):
        return all(self.in_bounds(x, lb, ub) for x in xs)
    
    def printn(self, msg: str) -> None:
        print("\n" + "=" * 50)
        print(msg)
        print("=" * 50)



