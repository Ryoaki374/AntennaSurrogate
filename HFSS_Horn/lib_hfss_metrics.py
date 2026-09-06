"""Small, dependency-free reductions for HFSS far-field exports.

This module is intentionally compatible with the Python environment embedded in
AEDT 2023.2.  Keeping the numerical work here makes ``subprocess.py`` easier to
review while leaving all HFSS COM calls in that script.
"""

from __future__ import division

import csv
import math


C0 = 299792458.0


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
    """Load an HFSS Data Table export of re/im(rETheta)."""
    with open(path, "r") as csv_file:
        rows = list(csv.reader(csv_file))
    if len(rows) < 2:
        raise ValueError("rETheta export must contain a header and data rows")

    component_name = "re(rETheta)" if component == "re" else "im(rETheta)"
    headers = rows[0]

    frequency_column = None
    phi_column = None
    theta_column = None
    component_column = None
    for column_index, header in enumerate(headers):
        header_lower = header.lower()
        if header_lower.startswith("freq"):
            frequency_column = column_index
        elif header_lower.startswith("phi"):
            phi_column = column_index
        elif header_lower.startswith("theta"):
            theta_column = column_index
        if component_name.lower() in header_lower:
            component_column = column_index

    if None in (frequency_column, phi_column, theta_column, component_column):
        raise ValueError("Required rETheta columns were not found in {}".format(path))

    samples_by_frequency = {}
    for row in rows[1:]:
        if abs(float(row[phi_column]) - phi_deg) > 1.0e-10:
            continue
        frequency_ghz = float(row[frequency_column])
        samples_by_frequency.setdefault(frequency_ghz, []).append((
            float(row[theta_column]),
            float(row[component_column]),
        ))
    if not samples_by_frequency:
        raise ValueError("No {} rows found in {}".format(component_name, path))

    theta = None
    data = {}
    for frequency_ghz in sorted(samples_by_frequency):
        samples = sorted(samples_by_frequency[frequency_ghz])
        current_theta = [sample[0] for sample in samples]
        if theta is None:
            theta = current_theta
        elif current_theta != theta:
            raise ValueError("Theta samples differ between frequencies in {}".format(path))
        data[frequency_ghz] = [sample[1] for sample in samples]
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


