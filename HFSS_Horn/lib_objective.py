import csv
import cmath
import math
import os
import re
from collections.abc import Mapping

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit


SPEED_OF_LIGHT = 299792458.0
PHASE_CENTER_THETA_LIMIT_DEG = 10.0
PHASE_CENTER_Z_MIN_M = -10.0e-3
PHASE_CENTER_Z_MAX_M = 10.0e-3
PHASE_CENTER_Z_SAMPLES = 1001
ELLIPTICITY_F_NUMBER = 3.0
ELLIPTICITY_PUPIL_SAMPLES = 129
ELLIPTICITY_FFT_SIZE = 1024
ELLIPTICITY_FIT_THRESHOLD = 0.1


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
    if not all(math.isfinite(value) for value in list(real.values()) + list(imag.values())):
        return math.nan

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


def load_rel3x_csv(csv_path):
    """Load the quarter-plane complex rEL3X grid for each frequency."""
    with open(csv_path, newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.reader(csv_file)
        try:
            headers = [header.strip().lower() for header in next(reader)]
        except StopIteration as error:
            raise ValueError("rEL3X CSV must contain a header") from error

        def find_column(prefix):
            return next(
                (
                    index
                    for index, header in enumerate(headers)
                    if header.startswith(prefix)
                ),
                None,
            )

        frequency_index = find_column("freq")
        theta_index = find_column("theta")
        phi_index = find_column("phi")
        real_index = find_column("re(rel3x)")
        imag_index = find_column("im(rel3x)")
        required = (frequency_index, theta_index, phi_index, real_index, imag_index)
        if any(index is None for index in required):
            raise ValueError(
                "rEL3X CSV is missing a required frequency, angle, or field column"
            )

        samples_by_frequency = {}
        for row in reader:
            if not row:
                continue
            if len(row) <= max(required):
                raise ValueError("rEL3X CSV data row has fewer columns than its header")
            frequency_ghz = float(row[frequency_index])
            theta_deg = float(row[theta_index])
            phi_deg = float(row[phi_index])
            field = complex(float(row[real_index]), float(row[imag_index]))
            samples_by_frequency.setdefault(frequency_ghz, {})[
                (theta_deg, phi_deg)
            ] = field

    rel3x_by_frequency = {}
    for frequency_ghz in sorted(samples_by_frequency):
        samples = samples_by_frequency[frequency_ghz]
        theta_deg = np.array(sorted({point[0] for point in samples}), dtype=float)
        phi_deg = np.array(sorted({point[1] for point in samples}), dtype=float)
        expected_count = len(theta_deg) * len(phi_deg)
        if len(samples) != expected_count:
            raise ValueError(
                "rEL3X angular grid is incomplete at {:g} GHz".format(frequency_ghz)
            )
        rel3x_quarter = np.empty((len(theta_deg), len(phi_deg)), dtype=complex)
        for theta_position, theta_value in enumerate(theta_deg):
            for phi_position, phi_value in enumerate(phi_deg):
                rel3x_quarter[theta_position, phi_position] = samples[
                    (theta_value, phi_value)
                ]
        rel3x_by_frequency[frequency_ghz] = (
            theta_deg,
            phi_deg,
            rel3x_quarter,
        )
    if not rel3x_by_frequency:
        raise ValueError("rEL3X CSV contains no field samples")
    return rel3x_by_frequency


def quarter_rel3x_to_pupil(
    theta_deg,
    phi_deg,
    rel3x_quarter,
    f_number=ELLIPTICITY_F_NUMBER,
    pupil_samples=ELLIPTICITY_PUPIL_SAMPLES,
):
    """Mirror the quarter-plane field and truncate it at the F-number stop."""
    pupil_axis = np.linspace(-1.0, 1.0, pupil_samples)
    pupil_x, pupil_y = np.meshgrid(pupil_axis, pupil_axis, indexing="xy")
    pupil_radius = np.hypot(pupil_x, pupil_y)
    theta_query_deg = np.degrees(np.arctan(pupil_radius / (2.0 * f_number)))
    phi_query_deg = np.degrees(np.arctan2(np.abs(pupil_y), np.abs(pupil_x)))

    interpolator = RegularGridInterpolator(
        (theta_deg, phi_deg),
        rel3x_quarter,
        bounds_error=False,
        fill_value=0.0,
    )
    query_points = np.column_stack(
        (theta_query_deg.ravel(), phi_query_deg.ravel())
    )
    pupil_field = interpolator(query_points).reshape(pupil_x.shape)
    pupil_field[pupil_radius > 1.0] = 0.0
    return pupil_axis, pupil_field


def pupil_to_beam(pupil_axis, pupil_field, fft_size=ELLIPTICITY_FFT_SIZE):
    """Return normalized beam power from a zero-padded 2D pupil FFT."""
    pupil_samples = pupil_field.shape[0]
    if pupil_field.shape != (pupil_samples, pupil_samples):
        raise ValueError("pupil field must be square")
    if fft_size < pupil_samples:
        raise ValueError("FFT size must be at least the pupil sample count")

    padded = np.zeros((fft_size, fft_size), dtype=complex)
    start = fft_size // 2 - pupil_samples // 2
    padded[start:start + pupil_samples, start:start + pupil_samples] = pupil_field
    beam_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))
    beam_power = np.abs(beam_field) ** 2
    peak_power = beam_power.max()
    if not np.isfinite(peak_power) or peak_power <= 0.0:
        return np.array([], dtype=float), np.full_like(beam_power, np.nan, dtype=float)
    beam_power /= peak_power

    d_rho = pupil_axis[1] - pupil_axis[0]
    spatial_frequency = np.fft.fftshift(np.fft.fftfreq(fft_size, d=d_rho))
    beam_axis = 2.0 * spatial_frequency
    return beam_axis, beam_power


def _fwhm_1d(axis, profile):
    """Estimate one profile width for the Gaussian fit initial parameters."""
    peak_index = int(np.argmax(profile))
    half_power = 0.5 * profile[peak_index]
    left_candidates = np.where(profile[:peak_index] < half_power)[0]
    right_candidates = np.where(profile[peak_index:] < half_power)[0]
    if not len(left_candidates) or not len(right_candidates):
        raise ValueError("beam profile does not cross half power on both sides")
    left_index = left_candidates[-1]
    right_index = peak_index + right_candidates[0]
    left_crossing = np.interp(
        half_power,
        profile[left_index:left_index + 2],
        axis[left_index:left_index + 2],
    )
    right_crossing = np.interp(
        half_power,
        profile[right_index - 1:right_index + 1][::-1],
        axis[right_index - 1:right_index + 1][::-1],
    )
    return right_crossing - left_crossing


def rotated_gaussian(coordinates, amplitude, x0, y0, sigma_x, sigma_y, angle, offset):
    x, y = coordinates
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    x_rot = cos_angle * (x - x0) + sin_angle * (y - y0)
    y_rot = -sin_angle * (x - x0) + cos_angle * (y - y0)
    return offset + amplitude * np.exp(
        -0.5 * ((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2)
    )


def ellipticity_from_2d_gaussian(
    beam_axis,
    beam_power,
    fit_threshold=ELLIPTICITY_FIT_THRESHOLD,
):
    """Fit the main beam and return (major-minor)/(major+minor)."""
    beam_x, beam_y = np.meshgrid(beam_axis, beam_axis, indexing="xy")
    fit_mask = np.isfinite(beam_power) & (beam_power >= fit_threshold)
    if np.count_nonzero(fit_mask) < 7:
        return math.nan

    x_data = beam_x[fit_mask]
    y_data = beam_y[fit_mask]
    power_data = beam_power[fit_mask]
    peak_y, peak_x = np.unravel_index(np.nanargmax(beam_power), beam_power.shape)
    sigma_to_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))
    initial_parameters = (
        1.0,
        beam_axis[peak_x],
        beam_axis[peak_y],
        _fwhm_1d(beam_axis, beam_power[peak_y, :]) / sigma_to_fwhm,
        _fwhm_1d(beam_axis, beam_power[:, peak_x]) / sigma_to_fwhm,
        0.0,
        0.0,
    )
    parameters, _ = curve_fit(
        rotated_gaussian,
        (x_data, y_data),
        power_data,
        p0=initial_parameters,
        bounds=(
            (0.0, beam_axis.min(), beam_axis.min(), 1.0e-6, 1.0e-6, -np.pi / 2.0, -0.2),
            (2.0, beam_axis.max(), beam_axis.max(), np.inf, np.inf, np.pi / 2.0, 0.2),
        ),
        maxfev=50000,
    )
    sigma_x, sigma_y = parameters[3], parameters[4]
    major = max(sigma_x, sigma_y)
    minor = min(sigma_x, sigma_y)
    return float((major - minor) / (major + minor))


def calculate_beam_ellipticities(csv_path):
    """Return the fitted FFT beam ellipticity at every exported frequency."""
    results = {}
    for frequency_ghz, field_data in load_rel3x_csv(csv_path).items():
        theta_deg, phi_deg, rel3x_quarter = field_data
        pupil_axis, pupil_field = quarter_rel3x_to_pupil(
            theta_deg,
            phi_deg,
            rel3x_quarter,
        )
        beam_axis, beam_power = pupil_to_beam(pupil_axis, pupil_field)
        if beam_axis.size == 0:
            results[frequency_ghz] = math.nan
        else:
            results[frequency_ghz] = ellipticity_from_2d_gaussian(
                beam_axis,
                beam_power,
            )
    return results


def read_temp_output(csv_path, output_name):
    """Reduce an HFSS CSV export to the scalar used by the optimizer.

    S11 and boresight use their worst (maximum) in-band dB values. Crosspol uses its band
    average. Ellipticity is calculated at each frequency by applying the F#
    pupil stop to complex rEL3X, taking its 2D FFT, and fitting the normalized
    main beam with a rotated 2D Gaussian. Those frequency values are reduced
    to their mean. The phase-center report contains the best z at each
    frequency and is reduced to its population standard deviation.
    """
    if output_name == "ellipticity":
        ellipticities_by_frequency = calculate_beam_ellipticities(csv_path)
        ellipticities = [
            ellipticities_by_frequency[frequency]
            for frequency in sorted(ellipticities_by_frequency)
        ]
        if not ellipticities:
            raise ValueError("ellipticity CSV contains no frequency samples")
        return sum(ellipticities) / len(ellipticities)

    with open(csv_path, newline="", encoding="utf-8-sig") as csv_file:
        rows = list(csv.reader(csv_file))
    if len(rows) < 2:
        raise ValueError("HFSS output CSV must contain a header and at least one data row")

    if output_name == "phase_center":
        return calculate_phase_center_stability(csv_path)

    if output_name == "phasecenter":
        return _population_std([float(row[1]) for row in rows[1:]])

    if output_name in ("S11", "Crosspol", "boresight"):
        values = [float(row[-1]) for row in rows[1:]]
        if not all(math.isfinite(value) for value in values):
            return math.nan
        if output_name in ("S11", "boresight"):
            return max(values)
        return sum(values) / len(values)

    raise ValueError("unsupported HFSS output: {}".format(output_name))


def _get_field(value, name):
    if isinstance(value, Mapping):
        return value[name]
    return getattr(value, name)


def _get_optional_field(value, name, default):
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def replace_nonfinite_objectives(values, objective_config):
    """Replace NaN final objective outputs with their configured limits."""
    terms = _get_field(objective_config, "terms")
    limits = {
        _get_field(term, "column"): float(_get_field(term, "limit"))
        for term in terms
    }
    if set(values) != set(limits):
        raise ValueError(
            "objective outputs and configured columns differ: outputs={}, configured={}".format(
                sorted(values), sorted(limits)
            )
        )
    if not all(math.isfinite(limit) for limit in limits.values()):
        raise ValueError("objective limits must be finite")

    replaced = {}
    for column, value in values.items():
        numeric_value = float(value)
        replaced[column] = limits[column] if math.isnan(numeric_value) else numeric_value
    return replaced


def normalize_objective(value, target, limit):
    """Map target to zero and limit to one without clamping either side."""
    value = float(value)
    target = float(target)
    limit = float(limit)
    if not all(math.isfinite(item) for item in (value, target, limit)):
        raise ValueError("objective values, targets, and limits must be finite")
    if target == limit:
        raise ValueError("objective target and limit must differ")
    return (value - target) / (limit - target)


def calculate_lp_fom(values, objective_config, p=None):
    """Return the configured weighted objective from scalar outputs.

    ``aggregation = "lp"`` retains the legacy one-sided Lp violation norm.
    ``aggregation = "signed_l2"`` subtracts a weighted L2 reward for values
    better than target from the weighted L2 norm of target violations.
    """
    terms = _get_field(objective_config, "terms")
    values = replace_nonfinite_objectives(values, objective_config)
    if p is None:
        p = _get_field(objective_config, "p")
    p = float(p)
    if not math.isfinite(p) or p < 1.0:
        raise ValueError("objective p must be finite and at least one")

    aggregation = str(_get_optional_field(objective_config, "aggregation", "lp")).lower()
    if aggregation not in {"lp", "signed_l2"}:
        raise ValueError("objective aggregation must be 'lp' or 'signed_l2'")
    if aggregation == "signed_l2" and p != 2.0:
        raise ValueError("signed_l2 aggregation requires p=2")

    reward_weight = float(_get_optional_field(objective_config, "reward_weight", 0.25))
    if not math.isfinite(reward_weight) or reward_weight < 0.0:
        raise ValueError("objective reward_weight must be finite and non-negative")

    weighted_bad = 0.0
    weighted_good = 0.0
    weight_sum = 0.0
    for term in terms:
        column = _get_field(term, "column")
        weight = float(_get_field(term, "weight"))
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("objective weights must be finite and non-negative")
        normalized = normalize_objective(
            values[column], _get_field(term, "target"), _get_field(term, "limit")
        )
        bad = max(normalized, 0.0)
        weighted_bad += weight * bad ** p
        if aggregation == "signed_l2":
            good = max(-normalized, 0.0)
            weighted_good += weight * good ** 2
        weight_sum += weight

    if weight_sum <= 0.0:
        raise ValueError("sum of objective weights must be greater than zero")
    bad_lp = (weighted_bad / weight_sum) ** (1.0 / p)
    if aggregation == "lp":
        return bad_lp

    good_l2 = math.sqrt(weighted_good / weight_sum)
    return bad_lp - reward_weight * good_l2

