import csv
import math
from collections.abc import Mapping


def read_temp_output(csv_path, output_name):
    """Reduce an HFSS CSV export to the scalar used by the optimizer.

    Scalar reports (currently S11 and XPD) use the mean of their last data
    column.  The ellipticity report contains frequency followed by the Phi=0
    and Phi=90 half-power beam widths; its per-frequency ellipticity is
    (Phi90 - Phi0) / (Phi90 + Phi0).
    """
    with open(csv_path, newline="", encoding="utf-8-sig") as csv_file:
        rows = list(csv.reader(csv_file))
    if len(rows) < 2:
        raise ValueError("HFSS output CSV must contain a header and at least one data row")

    if output_name != "ellipticity":
        values = [float(row[-1]) for row in rows[1:]]
        return sum(values) / len(values)

    if len(rows[0]) < 3:
        raise ValueError("ellipticity CSV must contain frequency, Phi=0, and Phi=90 columns")
    ellipticities = []
    for row in rows[1:]:
        phi_0, phi_90 = float(row[1]), float(row[2])
        denominator = phi_90 + phi_0
        if denominator == 0:
            raise ValueError("ellipticity is undefined when Phi=0 and Phi=90 widths sum to zero")
        ellipticities.append((phi_90 - phi_0) / denominator)
    return sum(ellipticities) / len(ellipticities)


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
