#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Generated from HFSS_Horn/main_robustLCB_penalty.ipynb.
# Jupyter code-cell boundaries are retained as # %% markers.

# %%
from pathlib import Path
import sys
import importlib

_CWD = Path.cwd().resolve()
if (_CWD / "HFSS_Horn").is_dir():
    _REPO_DIR = _CWD
    _HORN_DIR = _CWD / "HFSS_Horn"
else:
    _HORN_DIR = _CWD
    _REPO_DIR = _CWD.parent

for _path in (_REPO_DIR, _HORN_DIR):
    _path_str = str(_path)
    if _path.exists() and _path_str not in sys.path:
        sys.path.insert(0, _path_str)

import lib_config as config

import os
import json
import numpy as np
import pandas as pd

import lib_gp
# Objective baseline uses only GaussianProcess. Other searchers are intentionally
# left unmodified and commented out for this HFSS_Horn workflow.
#import lib_grad
#import lib_grad_db
#import lib_random
#import lib_ga
#import lib_ga_db
import lib_backbone
lib_backbone = importlib.reload(lib_backbone)
import lib_plot as plot

# design
import lib_RFdesign

# IPython-only notebook helper omitted: %load_ext autoreload
# IPython-only notebook helper omitted: %autoreload 2

# %%
config_path = _HORN_DIR / "_config.toml"
if not config_path.exists():
    raise FileNotFoundError(f"HFSS Horn config not found: {config_path}")

_config = config._loadConfig(config_path)
app_config = config.initParams(_config, debug=True)

# %%
backbone = lib_backbone.Backbone(config = app_config)
gp = lib_gp.GaussianProcess(config = app_config)

def build_optimizer(method, config):
    if method == "gp":
        return lib_gp.GaussianProcess(config=config)
    # Objective baseline intentionally does not refactor or enable random / grad / GA searchers.
    # if method == "gradient":
    #     return lib_grad.GradientSearch(config=config)
    # if method == "gradient_db":
    #     return lib_grad_db.GradientSearch(config=config)
    # if method == "random":
    #     return lib_random.RandomSearch(config=config)
    # if method == "ga":
    #     return lib_ga.RealCodedGA(config=config)
    # if method == "ga_db":
    #     return lib_ga_db.RealCodedGA(config=config)
    raise ValueError(f"Unknown optimizer method for Objective baseline: {method}")

base_dir = app_config.env.dir_base
backbone.initStorer()

# %%
model_paths, model_paths_str = backbone._get_path_models()

# %%
model_paths

# %%
design = lib_RFdesign.ConvexHorn(model_path=model_paths[0])
step_info = design.genHornPoly(
    d_aperture=11.6,  # = pixel size [mm]
    d_middle=6.4,  # intermediate "pinch" diameter [mm] (~0.55 * Dap)
    d_waveguide=1.80,  # circular WG for ~140 GHz (TE11 cutoff ~97.6 GHz)
    total_length=20.0,  # [mm]
    # bottom -> top length fractions:
    # (waveguide, S-taper1, middle, S-taper2, aperture), traced from reference
    section_fracs=(0.18, 0.27, 0.21, 0.20, 0.14),
    n_pts_z=app_config.hfss.n_pts,
    n_pts_c=24,
)
design.plotProfile2D(step_info)
design.plotHornPoly3D(step_info, n_pts_c=24)

# %%
ROUND_DECIMALS = app_config.runtime.round_decimals
RESULTS_FILE = str(backbone._get_dir_run() / Path(_config["io"]["filename_output"]))
LEGACY_TEMP_FILE = str(backbone._get_dir_run() / Path(_config["io"].get("filename_temp", "temp_hfss_export.csv")))
OBJECTIVE_COL = app_config.objective.name
TEMP_OUTPUTS = [
    {
        "name": output.name,
        "path": backbone._get_dir_run() / Path(output.filename),
    }
    for output in app_config.io.temp_outputs
]
if not TEMP_OUTPUTS:
    TEMP_OUTPUTS = [{"name": "S11", "path": Path(LEGACY_TEMP_FILE)}]

# %%
from lib_objective import calculate_lp_fom, normalize_objective

OBJECTIVE_P = getattr(app_config.objective, "p", 2.0)

# %%
# FoM normalization and weighted L_p validation tests
print("Case 1:", normalize_objective(20, 10, 30))
assert np.isclose(normalize_objective(20, 10, 30), 0.5)

print("Case 2:", normalize_objective(80, 100, 60))
assert np.isclose(normalize_objective(80, 100, 60), 0.5)

print("Case 3 minimization better than target:", normalize_objective(5, 10, 30))
print("Case 3 maximization better than target:", normalize_objective(120, 100, 60))
assert np.isclose(normalize_objective(5, 10, 30), -0.25)
assert np.isclose(normalize_objective(120, 100, 60), -0.5)

print("Case 4:", normalize_objective(40, 10, 30))
assert np.isclose(normalize_objective(40, 10, 30), 1.5)

case5_unbalanced_cfg = {"terms": [
    {"column": "a", "weight": 1.0, "target": 0.0, "limit": 1.0},
    {"column": "b", "weight": 1.0, "target": 0.0, "limit": 1.0},
]}
case5_unbalanced = calculate_lp_fom({"a": 0.1, "b": 0.9}, case5_unbalanced_cfg, p=2.0)
case5_balanced = calculate_lp_fom({"a": 0.5, "b": 0.5}, case5_unbalanced_cfg, p=2.0)
print("Case 5 unbalanced:", case5_unbalanced)
print("Case 5 balanced:", case5_balanced)
assert np.isclose(case5_unbalanced, np.sqrt(0.41))
assert np.isclose(case5_balanced, 0.5)
assert case5_unbalanced > case5_balanced

case6_cfg = {"terms": [
    {"column": "minimize", "weight": 1.0, "target": 10.0, "limit": 30.0},
    {"column": "maximize", "weight": 1.0, "target": 100.0, "limit": 60.0},
]}
case6 = calculate_lp_fom({"minimize": 20.0, "maximize": 80.0}, case6_cfg, p=2.0)
print("Case 6:", case6)
assert np.isclose(case6, 0.5)

error_cases = [
    ("target == limit", lambda: normalize_objective(1.0, 1.0, 1.0)),
    ("p < 1", lambda: calculate_lp_fom({"a": 0.0}, {"terms": [{"column": "a", "weight": 1.0, "target": 0.0, "limit": 1.0}]}, p=0.5)),
    ("negative weight", lambda: calculate_lp_fom({"a": 0.0}, {"terms": [{"column": "a", "weight": -1.0, "target": 0.0, "limit": 1.0}]}, p=2.0)),
    ("all weights zero", lambda: calculate_lp_fom({"a": 0.0}, {"terms": [{"column": "a", "weight": 0.0, "target": 0.0, "limit": 1.0}]}, p=2.0)),
    ("missing configured quantity", lambda: calculate_lp_fom({"missing": 0.0}, {"terms": [{"column": "a", "weight": 1.0, "target": 0.0, "limit": 1.0}]}, p=2.0)),
]
for label, func in error_cases:
    try:
        func()
    except ValueError as exc:
        print(f"Case 7 {label}: raised {exc}")
    else:
        raise AssertionError(f"Case 7 {label}: expected ValueError")

print("All FoM tests passed.")

# %%
import time

def wait_for_temp_outputs(temp_outputs, timeout_sec=600, poll_interval_sec=1.0):
    start = time.time()
    paths = [Path(item["path"]) for item in temp_outputs]
    while True:
        if all(path.exists() for path in paths):
            return
        if time.time() - start > timeout_sec:
            raise TimeoutError(f"Timed out waiting for temp outputs: {paths}")
        time.sleep(poll_interval_sec)


def read_temp_output_like_current_s11(temp_hfss_path, output_name):
    from lib_objective import read_temp_output

    return read_temp_output(temp_hfss_path, output_name)


def read_all_temp_outputs(temp_outputs, objective_cfg):
    from lib_objective import replace_nonfinite_objectives

    outputs = {}
    for item in temp_outputs:
        outputs[item["name"]] = read_temp_output_like_current_s11(item["path"], item["name"])
    return replace_nonfinite_objectives(outputs, objective_cfg)


def compute_objective(outputs, objective_cfg):
    return calculate_lp_fom(outputs, objective_cfg, p=getattr(objective_cfg, "p", 2.0))


def round_numeric_row(row, decimals=ROUND_DECIMALS):
    rounded = dict(row)
    for key, value in list(rounded.items()):
        if pd.notna(value) and isinstance(value, (int, float, np.integer, np.floating)):
            rounded[key] = float(np.round(value, decimals))
    return rounded


def getResult(input_params, param_names, temp_outputs, result_file_path):
    wait_for_temp_outputs(temp_outputs)
    header_flag = not os.path.exists(result_file_path)

    outputs = read_all_temp_outputs(temp_outputs, app_config.objective)
    result_row = dict(zip(param_names, input_params))
    result_row.update(outputs)
    result_row[OBJECTIVE_COL] = compute_objective(outputs, app_config.objective)
    result_row = round_numeric_row(result_row)

    df_result = pd.DataFrame([result_row])
    df_result.to_csv(result_file_path, mode='a', header=header_flag, index=False)

    for item in temp_outputs:
        try:
            os.remove(item["path"])
        except OSError:
            pass
    return True

# %%
ROUND_DECIMALS = app_config.runtime.round_decimals
SECTION_FRAC_SLICE = slice(2, 7)
SECTION_FRAC_NAMES = ("f_wg", "f_t1", "f_mid", "f_t2", "f_ap")
SECTION_FRAC_TOL = 1.0e-8


def section_frac_sum_constraint(x):
    """
    Equality constraint:
        f_wg + f_t1 + f_mid + f_t2 + f_ap = 1.0
    """
    f_wg, f_t1, f_mid, f_t2, f_ap = np.asarray(x, dtype=float).flatten()[SECTION_FRAC_SLICE]
    return f_wg + f_t1 + f_mid + f_t2 + f_ap - 1.0


def enforce_horn_constraints(params, decimals=ROUND_DECIMALS):
    """Project horn section fractions onto the bounded sum-to-one simplex."""
    return lib_backbone.enforce_section_frac_constraint(
        params,
        lower_bounds=cfg.lower_bounds,
        upper_bounds=cfg.upper_bounds,
        decimals=decimals,
    )


def assert_horn_constraints(params, tol=SECTION_FRAC_TOL):
    residual = section_frac_sum_constraint(params)
    if abs(residual) > tol:
        raise ValueError(f"section_fracs must sum to 1.0; residual={residual}")
    return True

# --- A. HFSS ---
def target_hfss(_config_temp, sim_id, param_names, params):
    params = enforce_horn_constraints(params)
    assert_horn_constraints(params)
    backbone.call_subroutine(_config_temp, sim_id, param_names, params, value_fmt=f"{{:.{ROUND_DECIMALS}f}}")
    getResult(params, param_names, TEMP_OUTPUTS, RESULTS_FILE)
    df_result = pd.read_csv(RESULTS_FILE)
    return df_result.iloc[-1][OBJECTIVE_COL]

# --- B. Synthetic Test Function ---
def target_ackley(params):
    
    # --- 2. Calculation ---
    x = np.array(params)
    n = len(params)

    arg1 = -0.2 * np.sqrt((1.0/n) * np.sum(x**2))
    arg2 = (1.0/n) * np.sum(np.cos(2. * np.pi * x))
    
    y = -20. * np.exp(arg1) - np.exp(arg2) + 20. + np.e

    return y

def target_add(params):

    x = np.array(params)

    y = np.sum(params)

    return y

def target_griewank(params):
    
    x = np.array(params)
    n = len(params)

    sum_term = np.sum(x**2) / 4000.0

    indices = np.arange(1, n + 1)
    prod_term = np.prod(np.cos(x / np.sqrt(indices)))
    
    y = 1.0 + sum_term - prod_term
    
    return y

costFunction = target_add
costFunction = target_ackley
costFunction = target_griewank
costFunction = target_hfss

# =====================================
if costFunction == target_hfss:
    cfg = app_config.hfss
elif costFunction == target_ackley:
    cfg = app_config.test
elif costFunction == target_add:
    cfg = app_config.test
elif costFunction == target_griewank:
    cfg = app_config.test


def round_params(params, decimals=ROUND_DECIMALS):
    params = np.asarray(params, dtype=float).flatten()
    if costFunction == target_hfss:
        params = enforce_horn_constraints(params, decimals=decimals)
    return np.round(params, decimals)


def round_history_row(row, param_names, decimals=ROUND_DECIMALS):
    return round_numeric_row(row, decimals=decimals)


LOWER_BOUNDS = cfg.lower_bounds
UPPER_BOUNDS = cfg.upper_bounds
PARAM_NAMES = cfg.param_names
N_REPT = cfg.n_repeats
N_INIT = cfg.n_init
N_SIML = cfg.n_simulation

# %%
if cfg.n_simulation <= cfg.n_init:
    raise ValueError("n_simulation must be greater than n_init.")

# %%
backbone._get_path_models()[0]

# %%
_config_temp = {
"n_simulation": cfg.n_simulation,
"n_repeats": cfg.n_repeats,
#"param_names": cfg.param_names[:4],
#"param_units": cfg.param_units[:4],
"WATCH_DIR": str(backbone._get_dir_run()),
"INPUT_FILE": str(backbone._get_dir_run() / Path(_config["io"]["filename_input"])),
"MODEL_FILE": model_paths_str,
"RESULTS_FILE": str(backbone._get_dir_run() / Path(_config["io"]["filename_output"])),
"TEMP_FILE": str(Path(LEGACY_TEMP_FILE)),
"TEMP_OUTPUTS": [{"name": item["name"], "path": str(item["path"])} for item in TEMP_OUTPUTS],
"DONE_FLAG_FILE": str(backbone._get_dir_run() / Path("hfss.done")),
}

done_flag_path = Path(_config_temp["DONE_FLAG_FILE"])
done_flag_path.unlink(missing_ok=True)

with open(base_dir / Path("_config_HFSS.json"), 'w') as f:
        json.dump(_config_temp , f, indent=4)

print(f"Temporarily updated {base_dir / Path('_config_HFSS.json')} with run-specific WATCH_DIR for HFSS.")

# %%

def costFunctionWrapper(param_names, params,):

    params = round_params(params, decimals=ROUND_DECIMALS)
    if costFunction == target_hfss:
        assert_horn_constraints(params)
    sim_id = backbone._getSimulationID()

    y = costFunction(_config_temp, sim_id, param_names, params,)

    if costFunction == target_hfss:
        df_result = pd.read_csv(RESULTS_FILE)
        _newline = df_result.iloc[-1].to_dict()
    else:
        _newline = dict(zip(param_names, params))
        _newline[OBJECTIVE_COL] = float(np.round(y, ROUND_DECIMALS))

    _newline = round_history_row(_newline, param_names)

    return y, _newline

# %%
# Optimizer
# NOTE: random / gradient / GA searchers are intentionally not refactored for the
# Objective baseline. This notebook uses SEARCH_METHOD = "gp" for Objective-based GP.

SEARCH_METHOD = "gp"
optimizer = build_optimizer(SEARCH_METHOD, app_config)

# Robust LCB on J(x) acquisition settings.
# Standard LCB can be restored in lib_gp.GaussianProcess.search by using the
# existing BoTorch LCB path instead of acq_params["name"] = "robust_lcb_penalty_on_j".
ROBUST_KAPPA = 2.0
ROBUST_N_PERTURB = 64
ROBUST_N_CANDIDATES = 4096
ROBUST_PERTURBATION_STD_RATIO = 0.01
ROBUST_PENALTY_WEIGHT = 1.0
ROBUST_RNG = np.random.default_rng(101)

_bounds_width = np.asarray(UPPER_BOUNDS, dtype=float) - np.asarray(LOWER_BOUNDS, dtype=float)
ROBUST_PERTURBATION_STD = ROBUST_PERTURBATION_STD_RATIO * _bounds_width
CUSTOM_PERTURBATION_STD = {
    "d_m": 0.1,
    "l_tot": 0.1,
    "f_wg": 0.01,
    "f_t1": 0.01,
    "f_mid": 0.01,
    "f_t2": 0.01,
    "f_ap": 0.01,
}
CUSTOM_PERTURBATION_STD_ARRAY = np.asarray([CUSTOM_PERTURBATION_STD[name] for name in PARAM_NAMES], dtype=float)
ROBUST_SIGMA = np.diag(CUSTOM_PERTURBATION_STD_ARRAY ** 2)
ROBUST_BOUNDS = np.vstack([np.asarray(LOWER_BOUNDS, dtype=float), np.asarray(UPPER_BOUNDS, dtype=float)]).T

# Robust LCB with posterior-mean variation penalty:
# acquisition = mu_J - kappa * sigma_J + penalty_weight * posterior_mu_sample_std
# posterior_mu_sample_std is the sample std of GP posterior mean mu_Z
# over perturbed inputs Z = x + delta.
# The perturbation range is controlled by CUSTOM_PERTURBATION_STD / ROBUST_SIGMA.
ROBUST_ACQ_PARAMS = {
    "name": "robust_lcb_penalty_on_j",
    "kappa": ROBUST_KAPPA,
    "penalty_weight": ROBUST_PENALTY_WEIGHT,
    "Sigma": ROBUST_SIGMA,
    "n_perturb": ROBUST_N_PERTURB,
    "n_candidates": ROBUST_N_CANDIDATES,
    "rng": ROBUST_RNG,
}

def _build_hidden_row(row, routine_idx, method_name):
    hidden_row = dict(row)
    hidden_row["routine_idx"] = routine_idx
    hidden_row["method"] = method_name
    hidden_row["visibility"] = "hidden"
    return round_history_row(hidden_row, PARAM_NAMES)


def _save_hidden_history(debug_rows, repeat_idx):
    if not debug_rows:
        return None

    df_debug = pd.DataFrame(debug_rows)
    csv_path = backbone._get_dir_run() / f"debug_repeat_{repeat_idx}.csv"
    df_debug.to_csv(csv_path, index=False)
    print(f"Saved hidden debug rows to: {csv_path}")
    return csv_path


def _append_visible_row(row, visible_history, routine_idx, info):
    row_visible = round_history_row(row, PARAM_NAMES)
    row_visible['Metric'] = float(np.round(info.get('acq', np.nan), ROUND_DECIMALS)) if pd.notna(info.get('acq', np.nan)) else np.nan
    row_visible['gamma'] = float(np.round(info.get('length_scale', np.nan), ROUND_DECIMALS)) if pd.notna(info.get('length_scale', np.nan)) else np.nan
    row_visible['routine_idx'] = routine_idx
    visible_history.append(row_visible)


def optBySearch(n_simulation, history_data, visible_history, hidden_history, initial_rows, lower_bound, upper_bound, active_indices=None, fixed_point=None):

    initial_count = len(initial_rows)
    total_budget = max(0, n_simulation - initial_count)
    if total_budget == 0:
        print("[search budget] no remaining simulation budget after initialization")
        return

    if not initial_rows:
        print("[search budget] no seed rows available for optimizer search")
        return

    if SEARCH_METHOD == "gradient_db":
        seed_row = dict(min(initial_rows, key=lambda row: row[OBJECTIVE_COL]))
        seed_label = "best_initial"
        remaining_budget = total_budget
        routine_idx = 1

        #print(
        #    f"[search routine] entering {routine_idx}/1 "
        #    f"(remaining budget incl. this: {remaining_budget}, seed: {seed_label}, n_init: {initial_count})"
        #)

        x_new, info = optimizer.search(
            history_data=history_data,
            param_names=PARAM_NAMES,
            lower_bounds=lower_bound,
            upper_bounds=upper_bound,
            objective_func=costFunctionWrapper,
            acq_params=ROBUST_ACQ_PARAMS if SEARCH_METHOD == "gp" else None,
            active_indices=active_indices,
            fixed_point=fixed_point,
            routine_index=routine_idx,
            routine_total=1,
            start_row=seed_row,
            maxiter=max(1, remaining_budget),
            maxfun=remaining_budget,
            max_evals=remaining_budget,
        )

        if x_new is not None:
            x_log = round_params(x_new, decimals=ROUND_DECIMALS)
            pre_row = dict(zip(PARAM_NAMES, np.asarray(x_log, dtype=float).flatten()))
            pre_row[OBJECTIVE_COL] = np.nan
            pre_row["phase"] = "x_new_pre_eval"
            hidden_history.append(_build_hidden_row(pre_row, routine_idx, "pre_eval"))

        method_name = info.get('method', SEARCH_METHOD)
        evaluated_rows = info.get('evaluated_rows', [])
        if not evaluated_rows and x_new is not None:
            _, fallback_row = costFunctionWrapper(PARAM_NAMES, x_new)
            evaluated_rows = [fallback_row]

        for row in evaluated_rows:
            rounded_row = round_history_row(row, PARAM_NAMES)
            history_data.append(rounded_row)
            hidden_history.append(_build_hidden_row(rounded_row, routine_idx, method_name))
            _append_visible_row(rounded_row, visible_history, routine_idx, info)

        if not evaluated_rows:
            print("[search routine] stopped at 1/1 because the optimizer produced no new evaluations")
            return

        print("[search routine] exited 1/1 (remaining budget: 0)")
        print(f"{routine_idx:<5} | {float(np.round(info.get('acq', np.nan), ROUND_DECIMALS)):<10} | {method_name:<10} | {float(np.round(info.get('length_scale', np.nan), ROUND_DECIMALS)) if pd.notna(info.get('length_scale', np.nan)) else np.nan}")
        return

    routine_idx = 0
    while routine_idx < total_budget:
        used_budget = max(0, len(visible_history) - initial_count)
        remaining_budget = total_budget - used_budget
        if remaining_budget <= 0:
            print(f"[search budget] exhausted after routine {routine_idx}")
            break

        routine_idx += 1
        if routine_idx <= len(initial_rows):
            seed_row = dict(initial_rows[routine_idx - 1])
            seed_label = f"initial[{routine_idx - 1}]"
        else:
            seed_row = dict(min(history_data, key=lambda row: row[OBJECTIVE_COL]))
            seed_label = "best_so_far"

        #print(
        #    f"[search routine] entering {routine_idx}/{total_budget} "
        #    f"(remaining budget incl. this: {remaining_budget}, seed: {seed_label})"
        #)

        x_new, info = optimizer.search(
            history_data=history_data,
            param_names=PARAM_NAMES,
            lower_bounds=lower_bound,
            upper_bounds=upper_bound,
            objective_func=costFunctionWrapper,
            acq_params=ROBUST_ACQ_PARAMS if SEARCH_METHOD == "gp" else None,
            active_indices=active_indices,
            fixed_point=fixed_point,
            routine_index=routine_idx,
            routine_total=total_budget,
            start_row=seed_row,
            maxiter=max(1, remaining_budget),
            maxfun=remaining_budget,
            max_evals=remaining_budget,
        )

        # 追加: x_new を先に記録（評価結果がなくても残す）
        if x_new is not None:
            x_log = round_params(x_new, decimals=ROUND_DECIMALS)
            pre_row = dict(zip(PARAM_NAMES, np.asarray(x_log, dtype=float).flatten()))
            pre_row[OBJECTIVE_COL] = np.nan
            pre_row["phase"] = "x_new_pre_eval"
            hidden_history.append(_build_hidden_row(pre_row, routine_idx, "pre_eval"))

        method_name = info.get('method', SEARCH_METHOD)
        evaluated_rows = info.get('evaluated_rows', [])
        if not evaluated_rows and x_new is not None:
            _, fallback_row = costFunctionWrapper(PARAM_NAMES, x_new)
            evaluated_rows = [fallback_row]

        for row in evaluated_rows:
            rounded_row = round_history_row(row, PARAM_NAMES)
            history_data.append(rounded_row)
            hidden_history.append(_build_hidden_row(rounded_row, routine_idx, method_name))
            _append_visible_row(rounded_row, visible_history, routine_idx, info)

        if not evaluated_rows:
            print(
                f"[search routine] stopped at {routine_idx}/{total_budget} "
                "because the optimizer produced no new evaluations"
            )
            break

        print(
            f"[search routine] exited {routine_idx}/{total_budget} "
            f"(remaining budget: {total_budget - max(0, len(visible_history) - initial_count)})"
        )
        print(f"{routine_idx:<5} | {float(np.round(info.get('acq', np.nan), ROUND_DECIMALS)):<10} | {method_name:<10} | {float(np.round(info.get('length_scale', np.nan), ROUND_DECIMALS)) if pd.notna(info.get('length_scale', np.nan)) else np.nan}")

# %%
import time
#my_best_input = [
#    [6.4, 20.0, 0.18, 0.27, 0.21, 0.20, 0.14],
#]
#my_best_input = [enforce_horn_constraints(x).tolist() for x in my_best_input]
#for x in my_best_input:
#    assert_horn_constraints(x)

# backbone.all_in_bounds(my_best_input, cfg.lower_bounds, cfg.upper_bounds)
active_indices, fixed_point, _ = backbone._buildSamplingIndices(dims=cfg.n_params, param_groups=cfg.param_groups, group_order=getattr(cfg, "group_order", None),)
assert_horn_constraints(fixed_point)

print("Active Indices:", active_indices)
print("Fixed Point:", np.round(fixed_point, ROUND_DECIMALS))

try:
    
    for r in range(N_REPT):
        backbone.printn(f"Starting {SEARCH_METHOD} Repeat {r + 1}/{N_REPT}")
        start = time.perf_counter()

        history_data = []
        visible_history = []
        hidden_history = []
        initial_rows = []
        
        # -------------------- initial simulation --------------------------------
        backbone.printn(f"--- Generating {N_INIT} Initial Samples ---")
        
        X_initial = backbone.LHSsampler_extended(
            dims=cfg.n_params,
            nums=cfg.n_init,
            lower_bounds=cfg.lower_bounds,
            upper_bounds=cfg.upper_bounds,
            active_indices=active_indices,
            fixed_point=fixed_point,
            #fixed_points = my_best_input # you can also reduce the dim of  my best_input[Ninit:]
        )
        X_initial = np.vstack([enforce_horn_constraints(x) for x in X_initial])
        X_initial = np.round(X_initial, ROUND_DECIMALS)
        for x in X_initial:
            assert_horn_constraints(x)
        
        print(f"{'Iter':<5} | {'New y':<10} | {'Method':<10} | {'Metric'}")        
        for i in range(N_INIT):
            params = X_initial[i]

            # add evaluation points
            pre_row = dict(zip(PARAM_NAMES, np.asarray(params, dtype=float).flatten()))
            pre_row[OBJECTIVE_COL] = np.nan
            pre_row["phase"] = "init_pre_eval"
            hidden_history.append(_build_hidden_row(pre_row, 0, "pre_eval"))

            y_new, _newline = costFunctionWrapper(PARAM_NAMES, params,)
            _newline['Metric'] = np.nan
            _newline['gamma'] = np.nan
            _newline['routine_idx'] = 0
            _newline = round_history_row(_newline, PARAM_NAMES)
            history_data.append(_newline)
            visible_history.append(_newline)
            initial_rows.append(dict(_newline))


        # -------------------- Search --------------------------------
        optBySearch(
            N_SIML,
            history_data,
            visible_history,
            hidden_history,
            initial_rows,
            LOWER_BOUNDS,
            UPPER_BOUNDS,
            active_indices=active_indices,
            fixed_point=fixed_point,
        )

        # ==============================================================================
        elapsed = time.perf_counter() - start
        print(f"Repeat {i} completed in {elapsed:.3f} seconds.")
        
        df_final = pd.DataFrame(visible_history)
        X_train = df_final[PARAM_NAMES].values
        y_train = df_final[OBJECTIVE_COL].values

        best_idx_final = np.argmin(y_train)
        print("-" * 75)
        print(f"Optimization Finished.")
        print(f"Global Best Found: y = {float(np.round(y_train[best_idx_final], ROUND_DECIMALS)):.10f}")
        
        best_x_str = np.array2string(np.round(X_train[best_idx_final], ROUND_DECIMALS), precision=ROUND_DECIMALS, separator=', ')
        print(f"At location: x = {best_x_str}")
        
        # --- After each repeat, archive results and save plot data ---
        df_output = backbone._genOutputDataFrame(df_final, objective_col=OBJECTIVE_COL)
        df_output[PARAM_NAMES] = df_output[PARAM_NAMES].round(ROUND_DECIMALS)
        for output in app_config.io.temp_outputs:
            if output.name in df_output:
                df_output[output.name] = df_output[output.name].round(ROUND_DECIMALS)
        if OBJECTIVE_COL in df_output:
            df_output[OBJECTIVE_COL] = df_output[OBJECTIVE_COL].round(ROUND_DECIMALS)
        if 'Metric' in df_output:
            df_output['Metric'] = df_output['Metric'].round(ROUND_DECIMALS)
        if 'gamma' in df_output:
            df_output['gamma'] = df_output['gamma'].round(ROUND_DECIMALS)
        _save_hidden_history(hidden_history, r + 1)
        
        # save
        backbone._addNewDatasetToHDF(df_output, "output", f"repeat_{r+1}")
        
        # --- 5a. Visualize learning curve of the final model ---
        plot.plot_learning_curve(df_output, objective_col=OBJECTIVE_COL)
        
                
finally:
    done_flag_path = Path(_config_temp["DONE_FLAG_FILE"])
    done_flag_path.touch()

    json_file = base_dir / Path("_config_HFSS.json")
    json_file.unlink(missing_ok=True) # delite the json file
    if backbone.h5f:
        backbone.h5f.close()

