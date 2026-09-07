import csv
import cmath
import math
import os
import re
from collections.abc import Mapping


SPEED_OF_LIGHT = 299792458.0
PHASE_CENTER_THETA_LIMIT_DEG = 10.0
PHASE_CENTER_Z_MIN_M = -10.0e-3
PHASE_CENTER_Z_MAX_M = 10.0e-3
PHASE_CENTER_Z_SAMPLES = 1001


def _phase_center_imag_path(real_path):
    root, extension = os.path.splitext(os.fspath(real_path))
    return root + "_imag" + extension


def _unit_scale(header, units):
    header_lower = header.lower()
    for unit, scale in units:
        if unit in header_lower:
            return scale
    return 1.0


def _frequency_from_trace_header(header):
    """Extract an HFSS trace frequency such as ``Freq='80GHz'`` in hertz."""
    match = re.search(
        r"\bFreq\s*=\s*['\"]?\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s*"
        r"(GHz|MHz|kHz|Hz)\b",
        header,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError("phase-center trace header does not contain a frequency: {}".format(header))
    scale = {"ghz": 1e9, "mhz": 1e6, "khz": 1e3, "hz": 1.0}[match.group(2).lower()]
    return float(match.group(1)) * scale


def _read_far_field_component(csv_path):
    with open(csv_path, newline="", encoding="utf-8-sig") as csv_file:
        rows = list(csv.reader(csv_file))
    if len(rows) < 2 or len(rows[0]) < 2:
        raise ValueError("phase-center CSV must contain Theta and at least one frequency trace")

    headers = rows[0]
    theta_index = next(
        (i for i, value in enumerate(headers) if re.match(r"^\s*theta(?:\s*\[|\s*$)", value, re.I)),
        None,
    )
    if theta_index is None:
        raise ValueError("phase-center CSV headers must identify Theta")
    theta_scale = _unit_scale(headers[theta_index], (("rad", 180.0 / math.pi), ("deg", 1.0)))

    # HFSS Data Table exports one Theta column and one trace column per frequency.
    trace_columns = [
        (index, _frequency_from_trace_header(header))
        for index, header in enumerate(headers)
        if index != theta_index
    ]
    frequencies = [frequency for _, frequency in trace_columns]
    if len(set(frequencies)) != len(frequencies):
        raise ValueError("phase-center CSV contains duplicate frequency trace columns")

    values = {}
    for row in rows[1:]:
        if not row or len(row) <= theta_index:
            continue
        theta_deg = float(row[theta_index]) * theta_scale
        for value_index, frequency in trace_columns:
            if len(row) <= value_index:
                raise ValueError("phase-center CSV data row has fewer columns than its header")
            values[(frequency, theta_deg)] = float(row[value_index])
    if not values:
        raise ValueError("phase-center CSV contains no numeric data rows")
    return values


def _unwrap(phases):
    unwrapped = [phases[0]]
    for phase in phases[1:]:
        delta = phase - unwrapped[-1]
        phase -= 2.0 * math.pi * math.floor((delta + math.pi) / (2.0 * math.pi))
        unwrapped.append(phase)
    return unwrapped


def calculate_phase_center_stability(real_csv_path):
    """Return the population STD of per-frequency brute-force phase centers in mm."""
    real = _read_far_field_component(real_csv_path)
    imag = _read_far_field_component(_phase_center_imag_path(real_csv_path))
    if set(real) != set(imag):
        raise ValueError("real and imaginary phase-center exports have different frequency/theta rows")

    by_frequency = {}
    for frequency, theta_deg in real:
        if abs(theta_deg) <= PHASE_CENTER_THETA_LIMIT_DEG + 1e-12:
            by_frequency.setdefault(frequency, []).append(theta_deg)
    if not by_frequency:
        raise ValueError("phase-center export has no samples in the configured main-beam angle")

    z_step = (PHASE_CENTER_Z_MAX_M - PHASE_CENTER_Z_MIN_M) / (PHASE_CENTER_Z_SAMPLES - 1)
    phase_centers_mm = []
    for frequency in sorted(by_frequency):
        theta_values = sorted(by_frequency[frequency])
        if len(theta_values) < 2:
            raise ValueError("phase-center fitting requires at least two theta samples per frequency")
        phases = _unwrap([
            cmath.phase(complex(real[(frequency, theta)], imag[(frequency, theta)]))
            for theta in theta_values
        ])
        cosines = [math.cos(math.radians(theta)) for theta in theta_values]
        wavenumber = 2.0 * math.pi * frequency / SPEED_OF_LIGHT

        best_z = None
        best_variance = None
        for index in range(PHASE_CENTER_Z_SAMPLES):
            z_value = PHASE_CENTER_Z_MIN_M + index * z_step
            corrected = [phase + wavenumber * z_value * cosine for phase, cosine in zip(phases, cosines)]
            mean_phase = sum(corrected) / len(corrected)
            variance = sum((phase - mean_phase) ** 2 for phase in corrected) / len(corrected)
            if best_variance is None or variance < best_variance:
                best_variance = variance
                best_z = z_value
        phase_centers_mm.append(best_z * 1e3)

    mean_center = sum(phase_centers_mm) / len(phase_centers_mm)
    variance = sum((center - mean_center) ** 2 for center in phase_centers_mm) / len(phase_centers_mm)
    return math.sqrt(variance)


def _population_std(values):
    """Return the population standard deviation of a non-empty sequence."""
    if not values:
        raise ValueError("population standard deviation requires at least one value")
    mean_value = sum(values) / len(values)
    return math.sqrt(sum((value - mean_value) ** 2 for value in values) / len(values))


def _ellipticities_from_rows(rows):
    """Return finite per-frequency ellipticities from an HFSS CSV export."""
    headers = [header.strip().lower() for header in rows[0]]
    phi_index = next(
        (
            index
            for index, header in enumerate(headers)
            if re.match(r"^phi(?:\\s*\\[|$)", header)
        ),
        None,
    )
    frequency_index = next(
        (index for index, header in enumerate(headers) if header.startswith("freq")),
        None,
    )

    # HFSS normally exports the report in long form:
    # Phi [deg], Freq [GHz], XWidthAtYVal(...) [deg].
    if phi_index is not None and frequency_index is not None:
        value_index = len(headers) - 1
        widths_by_frequency = {}
        invalid_frequencies = set()
        for row in rows[1:]:
            if len(row) <= max(phi_index, frequency_index, value_index):
                continue
            frequency = float(row[frequency_index])
            phi = float(row[phi_index])
            width = float(row[value_index])
            if not all(math.isfinite(value) for value in (frequency, phi, width)):
                invalid_frequencies.add(frequency)
                continue
            if abs(phi) <= 1.0e-10:
                phi_key = 0
            elif abs(phi - 90.0) <= 1.0e-10:
                phi_key = 90
            else:
                continue
            widths_by_frequency.setdefault(frequency, {})[phi_key] = width

        ellipticities = []
        for frequency in sorted(widths_by_frequency):
            widths = widths_by_frequency[frequency]
            if frequency in invalid_frequencies or 0 not in widths or 90 not in widths:
                continue
            denominator = widths[90] + widths[0]
            if denominator == 0:
                raise ValueError(
                    "ellipticity is undefined when Phi=0 and Phi=90 widths sum to zero"
                )
            ellipticities.append((widths[90] - widths[0]) / denominator)
        return ellipticities

    # Retain support for the earlier wide test/export format:
    # Freq, Phi=0 width, Phi=90 width.
    ellipticities = []
    for row in rows[1:]:
        phi_0, phi_90 = float(row[1]), float(row[2])
        if not math.isfinite(phi_0) or not math.isfinite(phi_90):
            continue
        denominator = phi_90 + phi_0
        if denominator == 0:
            raise ValueError("ellipticity is undefined when Phi=0 and Phi=90 widths sum to zero")
        ellipticities.append((phi_90 - phi_0) / denominator)
    return ellipticities


def read_temp_output(csv_path, output_name):
    """Reduce an HFSS CSV export to the scalar used by the optimizer.

    S11 uses the worst (maximum) in-band dB value. Crosspol uses its band
    average. The ellipticity report contains frequency followed by the Phi=0
    and Phi=90 half-power beam widths; its per-frequency ellipticity is
    (Phi90 - Phi0) / (Phi90 + Phi0), reduced to the population standard
    deviation over frequency. The phase-center report contains the best z at
    each frequency and likewise reduces to the population standard deviation
    of z over frequency.
    """
    with open(csv_path, newline="", encoding="utf-8-sig") as csv_file:
        rows = list(csv.reader(csv_file))
    if len(rows) < 2:
        raise ValueError("HFSS output CSV must contain a header and at least one data row")

    if output_name == "phase_center":
        return calculate_phase_center_stability(csv_path)

    if output_name == "phasecenter":
        return _population_std([float(row[1]) for row in rows[1:]])

    if output_name in ("S11", "Crosspol"):
        values = [float(row[-1]) for row in rows[1:]]
        if output_name == "S11":
            return max(values)
        return sum(values) / len(values)

    if output_name != "ellipticity":
        raise ValueError("unsupported HFSS output: {}".format(output_name))

    if len(rows[0]) < 3:
        raise ValueError("ellipticity CSV must contain frequency, Phi=0, and Phi=90 columns")
    ellipticities = _ellipticities_from_rows(rows)
    if not ellipticities:
        raise ValueError("ellipticity CSV contains no complete finite frequency samples")
    return _population_std(ellipticities)


def _get_field(value, name):
    if isinstance(value, Mapping):
        return value[name]
    return getattr(value, name)


def normalize_objective(value, target, limit):
    """Map target to zero and limit to one, clamping better values to zero."""
    value = float(value)
    target = float(target)
    limit = float(limit)
    if not all(math.isfinite(item) for item in (value, target, limit)):
        raise ValueError("objective values, targets, and limits must be finite")
    if target == limit:
        raise ValueError("objective target and limit must differ")
    return max(0.0, (value - target) / (limit - target))


def calculate_lp_fom(values, objective_config, p=None):
    """Return one weighted Lp objective from the configured scalar outputs."""
    terms = _get_field(objective_config, "terms")
    if p is None:
        p = _get_field(objective_config, "p")
    p = float(p)
    if not math.isfinite(p) or p < 1.0:
        raise ValueError("objective p must be finite and at least one")

    configured_columns = {_get_field(term, "column") for term in terms}
    if set(values) != configured_columns:
        raise ValueError(
            "objective outputs and configured columns differ: outputs={}, configured={}".format(
                sorted(values), sorted(configured_columns)
            )
        )

    weighted_sum = 0.0
    weight_sum = 0.0
    for term in terms:
        column = _get_field(term, "column")
        weight = float(_get_field(term, "weight"))
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("objective weights must be finite and non-negative")
        normalized = normalize_objective(
            values[column], _get_field(term, "target"), _get_field(term, "limit")
        )
        weighted_sum += weight * normalized ** p
        weight_sum += weight

    if weight_sum <= 0.0:
        raise ValueError("sum of objective weights must be greater than zero")
    return (weighted_sum / weight_sum) ** (1.0 / p)

