import csv
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib_hfss_metrics import C0, calculate_phase_centers, integrate_solid_angle


def test_integrated_crosspol_is_gain_l3y_fraction():
    theta = [-10.0, 0.0, 10.0]
    phi = [0.0, 45.0, 90.0]
    grid = {(theta_i, phi_i): (0.8, 0.2, 1.0) for theta_i in theta for phi_i in phi}

    cross = integrate_solid_angle(theta, phi, grid, 1)
    total = integrate_solid_angle(theta, phi, grid, 2)

    assert cross / total == pytest.approx(0.2)


def test_phasecenter_minimizes_unwrapped_phase_pkpk(tmp_path):
    frequency_ghz = 100.0
    expected_z_mm = 2.0
    wave_number = 2.0 * math.pi * frequency_ghz * 1.0e9 / C0
    theta_values = [-10.0, -5.0, 0.0, 5.0, 10.0]
    phases = [
        wave_number * expected_z_mm * 1.0e-3 * math.cos(math.radians(theta))
        for theta in theta_values
    ]

    re_path = tmp_path / "re.csv"
    im_path = tmp_path / "im.csv"
    for path, component, values in (
        (re_path, "re", [math.cos(phase) for phase in phases]),
        (im_path, "im", [math.sin(phase) for phase in phases]),
    ):
        with path.open("w", newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerow([
                "Theta [deg]",
                "{}(rETheta) [V] - Freq='100GHz' Phi='0deg'".format(component),
            ])
            writer.writerows(zip(theta_values, values))

    result = calculate_phase_centers(str(re_path), str(im_path))

    assert result[0][0] == 100.0
    assert result[0][1] == pytest.approx(expected_z_mm)
    assert result[0][2] == pytest.approx(0.0, abs=1.0e-10)

