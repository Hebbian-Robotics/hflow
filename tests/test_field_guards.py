"""Numeric guards reject invalid types before checking finite values and bounds."""

from collections.abc import Callable
from fractions import Fraction
from functools import partial

import pytest

from hflow._field_guards import (
    require_finite_float,
    require_int_in_range,
    require_non_negative_float,
    require_non_negative_int,
    require_non_negative_real,
    require_positive_float,
    require_positive_int,
)


@pytest.mark.parametrize(
    ("guard", "valid_values", "invalid_values"),
    [
        (require_positive_int, [1, 10**100], [0, -1, 1.0]),
        (require_non_negative_int, [0, 1, 10**100], [-1, 0.0]),
        (
            partial(require_int_in_range, minimum=0, maximum=2),
            [0, 1, 2],
            [-1, 3, 1.0],
        ),
        (require_finite_float, [-1, -0.5, 0, 0.0, 1, 1.5], []),
        (require_positive_float, [1, 0.5], [0, 0.0, -1, -0.5]),
        (require_non_negative_float, [0, 0.0, 1, 0.5], [-1, -0.5]),
        (
            require_non_negative_real,
            [0, 0.0, 1, 0.5, Fraction(1, 3)],
            [-1, -0.5, Fraction(-1, 3)],
        ),
    ],
    ids=[
        "positive-int",
        "non-negative-int",
        "int-range",
        "finite-float",
        "positive-float",
        "non-negative-float",
        "non-negative-real",
    ],
)
def test_numeric_guard_contract(
    guard: Callable[[object, str], object],
    valid_values: list[object],
    invalid_values: list[object],
) -> None:
    for value in valid_values:
        assert guard(value, "setting") is value
    for value in [
        *invalid_values,
        True,
        False,
        None,
        "1",
        1j,
        [],
        float("nan"),
        float("inf"),
        float("-inf"),
    ]:
        with pytest.raises(ValueError, match=r"^setting must be "):
            guard(value, "setting")
