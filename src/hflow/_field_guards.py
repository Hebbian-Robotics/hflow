"""Shared numeric-field type and range guards for caller-constructed settings.

Range checks alone (``0 <= x <= 100``) do not reject the wrong *type*: ``bool``
subclasses ``int``, so ``True``/``False`` satisfy every numeric comparison and
range test a settings dataclass runs, and a ``str``/``None`` would raise a
bare ``TypeError`` from the comparison itself instead of a clear message
naming the field. This closes that hole the same way ``catalog.py`` already
does for interval bounds and measurement values.

These guards are for caller-constructed configuration, not measurement-pipeline
output, so unlike ``catalog.py``'s NumPy-scalar coercion, no NumPy handling is
added here: a caller building a ``np.float64`` threshold can call ``.item()``
itself, the same way any other non-native-Python value would need to.

Range guards compose the type guards so callers can state the whole invariant
without repeating the bool exclusion. The Real guard preserves batching's
broader numeric contract without adding coercion to the native-number guards.
"""

import math
from numbers import Real
from typing import cast


def require_int(value: object, name: str) -> None:
    """Refuse anything but a plain ``int``, ``bool`` included."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int, got {type(value).__name__}")


def require_float(value: object, name: str) -> None:
    """Refuse anything but a plain ``int`` or ``float``, ``bool`` included.

    An ``int`` is accepted for a float-declared field: it is a perfectly good
    float value (``0`` is a real, falsy value that must still pass).
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be an int or float, got {type(value).__name__}")


def require_positive_int(value: object, name: str) -> None:
    """Refuse anything but a strictly positive int, excluding bool."""
    require_int(value, name)
    if cast(int, value) <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def require_non_negative_int(value: object, name: str) -> None:
    """Refuse anything but a non-negative int, excluding bool."""
    require_int(value, name)
    if cast(int, value) < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")


def require_int_in_range(value: object, name: str, *, minimum: int, maximum: int) -> None:
    """Refuse anything but an int within the inclusive bounds, excluding bool."""
    require_int(value, name)
    if not minimum <= cast(int, value) <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}], got {value}")


def require_finite_float(value: object, name: str) -> None:
    """Refuse anything but a finite int or float, excluding bool."""
    require_float(value, name)
    if not math.isfinite(cast(int | float, value)):
        raise ValueError(f"{name} must be finite, got {value}")


def require_positive_float(value: object, name: str) -> None:
    """Refuse anything but a finite, strictly positive int or float."""
    require_finite_float(value, name)
    if cast(int | float, value) <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def require_non_negative_float(value: object, name: str) -> None:
    """Refuse anything but a finite, non-negative int or float."""
    require_finite_float(value, name)
    if cast(int | float, value) < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")


def require_non_negative_real(value: object, name: str) -> None:
    """Preserve batching's finite, non-negative Real contract, excluding bool."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number, got {type(value).__name__}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
