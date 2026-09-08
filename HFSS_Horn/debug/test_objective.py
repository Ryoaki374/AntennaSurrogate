import sys
import math
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib_objective import (
    SPEED_OF_LIGHT,
    calculate_lp_fom,
    normalize_objective,
    read_temp_output,
    replace_nonfinite_objectives,
)


def test_normalize_handles_minimized_negative_db_values():
    assert normalize_objective(-30.0, -30.0, -10.0) == 0.0
    assert normalize_objective(-20.0, -30.0, -10.0) == 0.5
    assert normalize_objective(-40.0, -30.0, -10.0) == 0.0


def test_calculate_lp_fom_combines_s11_and_crosspol():
    config = SimpleNamespace(
        p=2.0,
        terms=[
            SimpleNamespace(column="S11", weight=1.0, target=-30.0, limit=-10.0),
            SimpleNamespace(column="Crosspol", weight=1.0, target=0.0, limit=0.1),
        ],
    )
    assert calculate_lp_fom({"S11": -20.0, "Crosspol": 0.05}, config) == pytest.approx(0.5)


def test_calculate_lp_fom_rejects_missing_outputs():
    config = {"p": 2.0, "terms": [{"column": "S11", "weight": 1.0, "target": -30, "limit": -10}]}
    with pytest.raises(ValueError, match="differ"):
        calculate_lp_fom({"Crosspol": 0.05}, config)


def test_calculate_lp_fom_accepts_an_explicit_p_override():
    config = {"terms": [{"column": "S11", "weight": 1.0, "target": -20, "limit": -10}]}
    assert calculate_lp_fom({"S11": -15.0}, config, p=2.0) == pytest.approx(0.5)


def test_active_hfss_outputs_match_objective_terms():
    config_path = Path(__file__).resolve().parents[1] / "_config.toml"
    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)

    output_names = [output["name"] for output in config["io"]["temp_outputs"]]
    objective_columns = [term["column"] for term in config["objective"]["terms"]]
    assert output_names == ["S11", "Crosspol", "ellipticity", "phasecenter"]
    assert objective_columns == output_names
    assert config["objective"]["terms"] == [
        {"column": "S11", "weight": 1.0, "target": -30.0, "limit": -20.0},
        {"column": "Crosspol", "weight": 1.0, "target": 0.01, "limit": 0.05},
        {"column": "ellipticity", "weight": 1.0, "target": 0.05, "limit": 0.2},
        {"column": "phasecenter", "weight": 1.0, "target": 2.0, "limit": 5.0},
    ]


def test_subprocess_uses_hfss_native_max_for_s11():
    subprocess_path = Path(__file__).resolve().parents[1] / "subprocess.py"
    source = subprocess_path.read_text(encoding="utf-8")
    assert '"y_component": "db(max(mag(S(Port1,Port1))))"' in source


def test_read_temp_output_calculates_phase_center_frequency_stability(tmp_path):
    real_export = tmp_path / "phase_center.csv"
    imag_export = tmp_path / "phase_center_imag.csv"
    headers = (
        '"Theta [deg]","re(rETheta) [V] - Freq=\'80GHz\' Phi=\'0deg\'",'
        '"re(rETheta) [V] - Freq=\'90GHz\' Phi=\'0deg\'"\n'
    )
    real_rows = []
    imag_rows = []
    frequencies_and_centers = ((80.0, -2.0), (90.0, 2.0))
    for theta_deg in (-10.0, -5.0, 0.0, 5.0, 10.0, 20.0):
        real_values = []
        imag_values = []
        for frequency_ghz, center_mm in frequencies_and_centers:
            wavenumber = 2.0 * math.pi * frequency_ghz * 1e9 / SPEED_OF_LIGHT
            # The out-of-window 20-degree point deliberately has unrelated phase.
            phase = -wavenumber * center_mm * 1e-3 * math.cos(math.radians(theta_deg))
            if theta_deg == 20.0:
                phase += 1.0
            real_values.append(math.cos(phase))
            imag_values.append(math.sin(phase))
        real_rows.append(",".join(map(str, (theta_deg, *real_values))) + "\n")
        imag_rows.append(",".join(map(str, (theta_deg, *imag_values))) + "\n")
    real_export.write_text(headers + "".join(real_rows), encoding="utf-8")
    imag_export.write_text(headers + "".join(imag_rows), encoding="utf-8")

    assert read_temp_output(real_export, "phase_center") == pytest.approx(2.0)


def test_phase_center_rejects_legacy_long_form_csv(tmp_path):
    real_export = tmp_path / "phase_center.csv"
    real_export.write_text(
        '"Freq [GHz]","Theta [deg]","field"\n80,-10,1\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="trace header does not contain a frequency"):
        read_temp_output(real_export, "phase_center")


def test_read_temp_output_uses_worst_in_band_s11(tmp_path):
    export = tmp_path / "s11.csv"
    export.write_text(
        "Freq,S11_dB\n80,-30\n90,-18\n100,-25\n",
        encoding="utf-8",
    )
    assert read_temp_output(export, "S11") == pytest.approx(-18.0)


def test_nan_s11_is_replaced_by_its_configured_limit(tmp_path):
    export = tmp_path / "s11.csv"
    export.write_text(
        "Freq,S11_dB\n80,-30\n90,nan\n100,-25\n",
        encoding="utf-8",
    )
    config = {
        "terms": [
            {"column": "S11", "weight": 1.0, "target": -30.0, "limit": -17.0}
        ]
    }

    raw_outputs = {"S11": read_temp_output(export, "S11")}
    assert math.isnan(raw_outputs["S11"])
    assert replace_nonfinite_objectives(raw_outputs, config) == {"S11": -17.0}


def test_read_temp_output_uses_crosspol_band_average(tmp_path):
    export = tmp_path / "crosspol.csv"
    export.write_text(
        "Freq,Crosspol\n80,0.01\n90,0.04\n100,0.10\n",
        encoding="utf-8",
    )
    assert read_temp_output(export, "Crosspol") == pytest.approx(0.05)


def test_read_temp_output_calculates_ellipticity_frequency_stability(tmp_path):
    export = tmp_path / "ellipticity.csv"
    export.write_text(
        '"Freq [GHz]","width - Phi=0","width - Phi=90"\n'
        "80,20,30\n"
        "81,30,30\n"
        "82,30,30\n",
        encoding="utf-8",
    )

    assert read_temp_output(export, "ellipticity") == pytest.approx(math.sqrt(2.0) / 15.0)


def test_nan_ellipticity_is_replaced_by_its_configured_limit(tmp_path):
    export = tmp_path / "ellipticity.csv"
    export.write_text(
        '"Phi [deg]","Freq [GHz]","XWidthAtYVal(GainTotal/PeakGain, 0.5) [deg]"\n'
        "0,80,20\n"
        "0,81,nan\n"
        "0,82,20\n"
        "90,80,30\n"
        "90,81,40\n"
        "90,82,20\n",
        encoding="utf-8",
    )

    raw_outputs = {"ellipticity": read_temp_output(export, "ellipticity")}
    config = {
        "terms": [
            {
                "column": "ellipticity",
                "weight": 1.0,
                "target": 0.05,
                "limit": 0.37,
            }
        ]
    }

    assert math.isnan(raw_outputs["ellipticity"])
    assert replace_nonfinite_objectives(raw_outputs, config) == {"ellipticity": 0.37}
    assert calculate_lp_fom(raw_outputs, config, p=2.0) == pytest.approx(1.0)


def test_replace_nonfinite_objectives_preserves_finite_values():
    config = {
        "terms": [
            {"column": "S11", "weight": 1.0, "target": -30.0, "limit": -20.0},
            {"column": "Crosspol", "weight": 1.0, "target": 0.01, "limit": 0.05},
        ]
    }

    assert replace_nonfinite_objectives(
        {"S11": -24.0, "Crosspol": float("inf")}, config
    ) == {"S11": -24.0, "Crosspol": 0.05}


def test_read_temp_output_rejects_zero_ellipticity_denominator(tmp_path):
    export = tmp_path / "ellipticity.csv"
    export.write_text("Freq,Phi0,Phi90\n80,-20,20\n", encoding="utf-8")

    with pytest.raises(ValueError, match="sum to zero"):
        read_temp_output(export, "ellipticity")


def test_read_temp_output_calculates_phasecenter_stability(tmp_path):
    export = tmp_path / "phasecenter.csv"
    export.write_text(
        "Frequency_GHz,PhaseCenterZ_mm,MinimumPhasePkPk_deg\n"
        "85,-1.0,2.0\n"
        "86,0.0,1.0\n"
        "87,1.0,2.0\n",
        encoding="utf-8",
    )

    assert read_temp_output(export, "phasecenter") == pytest.approx((2.0 / 3.0) ** 0.5)

