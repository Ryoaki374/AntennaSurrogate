"""Small, dependency-free reductions for HFSS far-field exports.

This module is intentionally compatible with the Python environment embedded in
AEDT 2023.2.  Keeping the numerical work here makes ``subprocess.py`` easier to
review while leaving all HFSS COM calls in that script.
"""

from __future__ import division

import csv
import math
import re


C0 = 299792458.0
FREQUENCY_PATTERN = re.compile(r"Freq='([0-9.]+)GHz'")


def numeric_values(start, stop, step):
    """Return an inclusive floating-point range."""
    count = int(round((stop - start) / step))
    return [start + index * step for index in range(count + 1)]


def unique_sorted(values):
    """Return sorted unique floats, tolerating insignificant COM duplicates."""
    result = []
    for value in sorted(float(item) for item in values):
        if not result or abs(value - result[-1]) > 1.0e-10:
            result.append(value)
    return result


def integrate_solid_angle(theta_values, phi_values, grid, component_index):
    """Integrate Gain*abs(sin(theta)) over a rectangular theta/phi grid."""
    theta_integrals = []
    for phi_value in phi_values:
        theta_integral = 0.0
        for index in range(1, len(theta_values)):
            theta0_deg = theta_values[index - 1]
            theta1_deg = theta_values[index]
            theta0 = math.radians(theta0_deg)
            theta1 = math.radians(theta1_deg)
            gain0 = grid[(theta0_deg, phi_value)][component_index]
            gain1 = grid[(theta1_deg, phi_value)][component_index]
            y0 = gain0 * abs(math.sin(theta0))
            y1 = gain1 * abs(math.sin(theta1))
            theta_integral += 0.5 * (y0 + y1) * (theta1 - theta0)
        theta_integrals.append(theta_integral)

    result = 0.0
    for index in range(1, len(phi_values)):
        phi0 = math.radians(phi_values[index - 1])
        phi1 = math.radians(phi_values[index])
        result += 0.5 * (
            theta_integrals[index - 1] + theta_integrals[index]
        ) * (phi1 - phi0)
    return result


def _load_retheta_table(path, component, phi_deg=0.0):
    """Load one HFSS wide-table export of re/im(rETheta)."""
    with open(path, "r") as csv_file:
        rows = list(csv.reader(csv_file))
    if len(rows) < 2:
        raise ValueError("rETheta export must contain a header and data rows")

    component_name = "re(rETheta)" if component == "re" else "im(rETheta)"
    phi_tag = "Phi='{:g}deg'".format(phi_deg)
    selected_columns = []
    for column_index, header in enumerate(rows[0][1:], 1):
        if component_name not in header:
            continue
        if "Phi=" in header and phi_tag not in header:
            continue
        match = FREQUENCY_PATTERN.search(header)
        if match:
            selected_columns.append((column_index, float(match.group(1))))
    if not selected_columns:
        raise ValueError("No {} columns found in {}".format(component_name, path))

    theta = [float(row[0]) for row in rows[1:]]
    data = {}
    for column_index, frequency_ghz in selected_columns:
        data[frequency_ghz] = [float(row[column_index]) for row in rows[1:]]
    return theta, data


def _unwrap(phases):
    """Unwrap radians using NumPy's default pi discontinuity convention."""
    if not phases:
        return []
    result = [phases[0]]
    offset = 0.0
    for index in range(1, len(phases)):
        delta = phases[index] - phases[index - 1]
        if delta > math.pi:
            offset -= 2.0 * math.pi
        elif delta < -math.pi:
            offset += 2.0 * math.pi
        result.append(phases[index] + offset)
    return result


def calculate_phase_centers(
    re_path,
    im_path,
    theta_min_deg=-10.0,
    theta_max_deg=10.0,
    z_min_mm=-30.0,
    z_max_mm=30.0,
    dz_mm=0.2,
    phase_sign=-1.0,
):
    """Find the z that minimizes phase peak-to-peak at every frequency."""
    theta_re, re_data = _load_retheta_table(re_path, "re")
    theta_im, im_data = _load_retheta_table(im_path, "im")
    if theta_re != theta_im:
        raise ValueError("Theta samples in re/im rETheta exports do not match")

    selected = [
        index for index, theta in enumerate(theta_re)
        if theta_min_deg <= theta <= theta_max_deg
    ]
    selected.sort(key=lambda index: theta_re[index])
    if len(selected) < 3:
        raise ValueError("Too few theta samples in the phase-center range")

    theta_rad = [math.radians(theta_re[index]) for index in selected]
    z_candidates_mm = numeric_values(z_min_mm, z_max_mm, dz_mm)
    frequencies = sorted(set(re_data).intersection(im_data))
    results = []

    for frequency_ghz in frequencies:
        phase0 = _unwrap([
            math.atan2(im_data[frequency_ghz][index], re_data[frequency_ghz][index])
            for index in selected
        ])
        wave_number = 2.0 * math.pi * frequency_ghz * 1.0e9 / C0

        best_z_mm = None
        best_pkpk_rad = None
        for z_mm in z_candidates_mm:
            z_m = z_mm * 1.0e-3
            corrected_phase = [
                phase + phase_sign * wave_number * z_m * math.cos(theta)
                for phase, theta in zip(phase0, theta_rad)
            ]
            phase_pkpk_rad = max(corrected_phase) - min(corrected_phase)
            if best_pkpk_rad is None or phase_pkpk_rad < best_pkpk_rad:
                best_z_mm = z_mm
                best_pkpk_rad = phase_pkpk_rad

        results.append((
            frequency_ghz,
            best_z_mm,
            math.degrees(best_pkpk_rad),
        ))
    return results

