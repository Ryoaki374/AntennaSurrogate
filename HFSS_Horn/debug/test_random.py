import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lib_random


def _objective(param_names, point):
    row = dict(zip(param_names, point))
    return float(np.sum(point)), row


def test_random_search_does_not_seed_lhs_from_routine_index(monkeypatch):
    seeds = []
    samples = iter((np.array([[0.1, 0.2]]), np.array([[0.7, 0.8]])))

    class RecordingLatinHypercube:
        def __init__(self, d, seed=None):
            seeds.append(seed)

        def random(self, n):
            return next(samples)

    monkeypatch.setattr(lib_random, "LatinHypercube", RecordingLatinHypercube)
    config = SimpleNamespace(
        runtime=SimpleNamespace(round_decimals=10),
        objective=SimpleNamespace(name="Objective"),
    )
    search = lib_random.RandomSearch(config)
    kwargs = dict(
        history_data=[],
        param_names=["x", "y"],
        lower_bounds=np.zeros(2),
        upper_bounds=np.ones(2),
        objective_func=_objective,
    )

    first, _ = search.search(routine_index=1, **kwargs)
    second, _ = search.search(routine_index=1, **kwargs)

    assert seeds == [None, None]
    assert not np.array_equal(first, second)


def test_random_search_keeps_inactive_dimensions_fixed(monkeypatch):
    class FixedLatinHypercube:
        def __init__(self, d, seed=None):
            assert d == 1
            assert seed is None

        def random(self, n):
            return np.array([[0.25]])

    monkeypatch.setattr(lib_random, "LatinHypercube", FixedLatinHypercube)
    config = SimpleNamespace(
        runtime=SimpleNamespace(round_decimals=10),
        objective=SimpleNamespace(name="Objective"),
    )
    search = lib_random.RandomSearch(config)

    point, info = search.search(
        history_data=[],
        param_names=["x", "y"],
        lower_bounds=np.array([0.0, 10.0]),
        upper_bounds=np.array([1.0, 20.0]),
        objective_func=_objective,
        active_indices=[1],
        fixed_point=np.array([0.4, 15.0]),
        routine_index=15,
    )

    assert point.tolist() == [0.4, 12.5]
    assert info["sampler"] == "lhs"
