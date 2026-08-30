import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib_objective import calculate_lp_fom, normalize_objective, read_temp_output


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


def test_calculate_lp_fom_accepts_an_explicit_p_override():
    config = {"terms": [{"column": "S11", "weight": 1.0, "target": -20, "limit": -10}]}
    assert calculate_lp_fom({"S11": -15.0}, config, p=2.0) == pytest.approx(0.5)


def test_active_hfss_outputs_match_objective_terms():
    config_path = Path(__file__).resolve().parents[1] / "_config.toml"
    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)

    output_names = [output["name"] for output in config["io"]["temp_outputs"]]
    objective_columns = [term["column"] for term in config["objective"]["terms"]]
    assert output_names == ["S11", "XPD", "ellipticity"]
    assert objective_columns == output_names


def test_read_temp_output_calculates_mean_ellipticity(tmp_path):
    export = tmp_path / "ellipticity.csv"
    export.write_text(
        '"Freq [GHz]","width - Phi=0","width - Phi=90"\n'
        "80,20,30\n"
        "81,30,30\n",
        encoding="utf-8",
    )

    assert read_temp_output(export, "ellipticity") == pytest.approx(0.1)


def test_read_temp_output_rejects_zero_ellipticity_denominator(tmp_path):
    export = tmp_path / "ellipticity.csv"
    export.write_text("Freq,Phi0,Phi90\n80,-20,20\n", encoding="utf-8")

    with pytest.raises(ValueError, match="sum to zero"):
        read_temp_output(export, "ellipticity")
