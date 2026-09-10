import sys
import math
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lib_objective
from lib_objective import (
    SPEED_OF_LIGHT,
    calculate_lp_fom,
    ellipticity_from_2d_gaussian,
    load_rel3x_csv,
    normalize_objective,
    pupil_to_beam,
    quarter_rel3x_to_pupil,
    read_temp_output,
    replace_nonfinite_objectives,
    rotated_gaussian,
)


def test_normalize_handles_minimized_negative_db_values():
    assert normalize_objective(-30.0, -30.0, -10.0) == 0.0
    assert normalize_objective(-20.0, -30.0, -10.0) == 0.5
    assert normalize_objective(-40.0, -30.0, -10.0) == -0.5


def test_signed_l2_rewards_values_better_than_target_without_clamping():
    config = SimpleNamespace(
        p=2.0,
        aggregation="signed_l2",
        reward_weight=0.25,
        terms=[
            SimpleNamespace(column="bad", weight=1.0, target=0.0, limit=1.0),
            SimpleNamespace(column="good", weight=1.0, target=0.0, limit=1.0),
        ],
    )

    result = calculate_lp_fom({"bad": 2.0, "good": -4.0}, config)
    assert result == pytest.approx(math.sqrt(2.0) - 0.25 * math.sqrt(8.0))


def test_signed_l2_is_zero_at_all_targets_and_one_at_all_limits():
    config = SimpleNamespace(
        p=2.0,
        aggregation="signed_l2",
        reward_weight=0.25,
        terms=[
            SimpleNamespace(column="a", weight=1.0, target=10.0, limit=30.0),
            SimpleNamespace(column="b", weight=1.0, target=100.0, limit=60.0),
        ],
    )

    assert calculate_lp_fom({"a": 10.0, "b": 100.0}, config) == pytest.approx(0.0)
    assert calculate_lp_fom({"a": 30.0, "b": 60.0}, config) == pytest.approx(1.0)


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
    assert config["objective"]["aggregation"] == "signed_l2"
    assert config["objective"]["reward_weight"] == pytest.approx(0.25)
    assert config["objective"]["terms"] == [
        {"column": "S11", "weight": 1.0, "target": -30.0, "limit": -20.0},
        {"column": "Crosspol", "weight": 1.0, "target": 0.01, "limit": 0.05},
        {"column": "ellipticity", "weight": 1.0, "target": 0.05, "limit": 0.2},
        {"column": "phasecenter", "weight": 1.0, "target": 2.0, "limit": 5.0},
    ]


def test_subprocess_uses_hfss_native_max_for_s11():
    subprocess_path = Path(__file__).resolve().parents[1] / "subprocess.py"
    source = subprocess_path.read_text(encoding="utf-8")
    assert '"y_component": "db(max(mag(S(Port1:1,Port1:1))))"' in source


def test_subprocess_exports_rel3x_instead_of_hfss_beam_width():
    subprocess_path = Path(__file__).resolve().parents[1] / "subprocess.py"
    source = subprocess_path.read_text(encoding="utf-8")
    assert 'GetRealDataValues("rEL3X", False)' in source
    assert 'GetImagDataValues("rEL3X", False)' in source
    assert "XWidthAtYVal" not in source
    assert "ELLIPTICITY_NAN_MODEL_FILENAME" not in source


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


def test_load_rel3x_csv_builds_complex_frequency_grids(tmp_path):
    export = tmp_path / "rel3x.csv"
    export.write_text(
        '"Freq [GHz]","Theta [deg]","Phi [deg]","re(rEL3X) [V]","im(rEL3X) [V]"\n'
        "85,0,0,1,2\n"
        "85,0,90,3,4\n"
        "85,1,0,5,6\n"
        "85,1,90,7,8\n",
        encoding="utf-8",
    )

    theta, phi, field = load_rel3x_csv(export)[85.0]
    assert theta.tolist() == [0.0, 1.0]
    assert phi.tolist() == [0.0, 90.0]
    np.testing.assert_array_equal(
        field,
        np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]]),
    )


def test_2d_gaussian_fit_returns_beam_ellipticity():
    beam_axis = np.linspace(-4.0, 4.0, 161)
    beam_x, beam_y = np.meshgrid(beam_axis, beam_axis, indexing="xy")
    beam_power = rotated_gaussian(
        (beam_x, beam_y),
        1.0,
        0.15,
        -0.2,
        1.2,
        0.8,
        0.35,
        0.0,
    )

    assert ellipticity_from_2d_gaussian(beam_axis, beam_power) == pytest.approx(0.2)


def test_circular_pupil_field_produces_circular_fitted_beam():
    theta = np.linspace(0.0, 9.5, 20)
    phi = np.linspace(0.0, 90.0, 19)
    field = np.ones((len(theta), len(phi)), dtype=complex)
    pupil_axis, pupil_field = quarter_rel3x_to_pupil(
        theta,
        phi,
        field,
        pupil_samples=65,
    )
    beam_axis, beam_power = pupil_to_beam(pupil_axis, pupil_field, fft_size=256)

    assert ellipticity_from_2d_gaussian(beam_axis, beam_power) == pytest.approx(
        0.0,
        abs=1.0e-6,
    )


def test_read_temp_output_calculates_fitted_ellipticity_frequency_stability(
    tmp_path,
    monkeypatch,
):
    export = tmp_path / "ellipticity.csv"
    export.write_text("header\nrow\n", encoding="utf-8")
    monkeypatch.setattr(
        lib_objective,
        "calculate_beam_ellipticities",
        lambda path: {85.0: 0.1, 87.0: 0.2, 89.0: 0.4},
    )

    assert read_temp_output(export, "ellipticity") == pytest.approx(
        np.std([0.1, 0.2, 0.4])
    )


def test_nan_final_ellipticity_is_replaced_by_its_configured_limit():
    raw_outputs = {"ellipticity": math.nan}
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

    assert replace_nonfinite_objectives(raw_outputs, config) == {"ellipticity": 0.37}
    assert calculate_lp_fom(raw_outputs, config, p=2.0) == pytest.approx(1.0)


def test_replace_nonfinite_objectives_preserves_non_nan_values():
    config = {
        "terms": [
            {"column": "S11", "weight": 1.0, "target": -30.0, "limit": -20.0},
            {"column": "Crosspol", "weight": 1.0, "target": 0.01, "limit": 0.05},
        ]
    }

    assert replace_nonfinite_objectives(
        {"S11": -24.0, "Crosspol": float("inf")}, config
    ) == {"S11": -24.0, "Crosspol": float("inf")}


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

