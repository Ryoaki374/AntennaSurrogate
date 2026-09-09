import sys
import subprocess  # Load stdlib before the repository's subprocess.py can shadow it.
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib_gp import GaussianProcess, _bounded_simplex_from_unit


PARAM_NAMES = ["d_m", "l_tot", "f_wg", "f_t1", "f_mid", "f_t2", "f_ap"]
LOWER = np.array([2.4, 10.0, 0.05, 0.05, 0.05, 0.05, 0.05])
UPPER = np.array([11.599999, 50.0, 0.999996, 0.999996, 0.999996, 0.999996, 0.999996])


def _config():
    return SimpleNamespace(
        hfss=SimpleNamespace(
            param_names=PARAM_NAMES,
            lower_bounds=LOWER.tolist(),
            upper_bounds=UPPER.tolist(),
        ),
        opt=SimpleNamespace(kernel_type="Matern52", length_scale=3.0),
    )


def test_stick_breaking_uses_four_coordinates_for_five_fractions():
    unit = np.array([[0.0, 0.0, 0.0, 0.0], [0.5, 0.25, 0.75, 1.0]])
    fractions = _bounded_simplex_from_unit(unit, LOWER[2:], UPPER[2:])

    assert fractions.shape == (2, 5)
    assert np.allclose(fractions.sum(axis=1), 1.0)
    assert np.all(fractions >= LOWER[2:] - 1.0e-12)
    assert np.all(fractions <= UPPER[2:] + 1.0e-12)


def test_gp_drops_dependent_fraction_and_generates_feasible_sobol_points():
    gp = GaussianProcess(_config())
    gp._configure_input_coordinates(PARAM_NAMES, LOWER, UPPER)

    candidates = gp.sample_sobol_candidates(
        np.column_stack([LOWER, UPPER]),
        n_candidates=4096,
        rng=np.random.default_rng(101),
    )
    encoded = gp._encode_model_inputs(candidates)

    assert candidates.shape == (4096, 7)
    assert encoded.shape == (4096, 6)
    assert np.allclose(candidates[:, 2:7].sum(axis=1), 1.0)
    assert np.all(candidates >= LOWER - 1.0e-12)
    assert np.all(candidates <= UPPER + 1.0e-12)
    assert np.unique(candidates, axis=0).shape[0] == 4096


def test_projection_keeps_robust_perturbations_on_simplex():
    gp = GaussianProcess(_config())
    gp._configure_input_coordinates(PARAM_NAMES, LOWER, UPPER)
    off_simplex = np.array([6.4, 20.0, 0.30, 0.30, 0.30, 0.30, 0.30])

    projected = gp.project_external_inputs(off_simplex)

    assert projected[2:7].sum() == pytest.approx(1.0)
    assert np.all(projected >= LOWER - 1.0e-12)
    assert np.all(projected <= UPPER + 1.0e-12)

