import math
from collections.abc import Mapping


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


def calculate_lp_fom(values, objective_config):
    """Return one weighted Lp objective from the configured scalar outputs."""
    terms = _get_field(objective_config, "terms")
    p = float(_get_field(objective_config, "p"))
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
