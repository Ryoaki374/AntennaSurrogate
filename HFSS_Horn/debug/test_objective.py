import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib_objective import calculate_lp_fom, normalize_objective


def test_normalize_handles_minimized_negative_db_values():
    assert normalize_objective(-30.0, -30.0, -10.0) == 0.0
    assert normalize_objective(-20.0, -30.0, -10.0) == 0.5
    assert normalize_objective(-40.0, -30.0, -10.0) == 0.0


def test_calculate_lp_fom_combines_s11_and_xpd():
    config = SimpleNamespace(
        p=2.0,
        terms=[
            SimpleNamespace(column="S11", weight=1.0, target=-30.0, limit=-10.0),
            SimpleNamespace(column="XPD", weight=1.0, target=-30.0, limit=-10.0),
        ],
    )
    assert calculate_lp_fom({"S11": -20.0, "XPD": -20.0}, config) == pytest.approx(0.5)


def test_calculate_lp_fom_rejects_missing_outputs():
    config = {"p": 2.0, "terms": [{"column": "S11", "weight": 1.0, "target": -30, "limit": -10}]}
    with pytest.raises(ValueError, match="differ"):
        calculate_lp_fom({"XPD": -20.0}, config)
