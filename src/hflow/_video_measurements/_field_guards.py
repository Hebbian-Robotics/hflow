"""Shared numeric-field validation for the video-measurement settings dataclasses.

Range checks alone (``0 <= x <= 100``) do not reject the wrong *type*: ``bool``
subclasses ``int``, so ``True``/``False`` satisfy every numeric comparison and
range test the settings below already run, and a ``str``/``None`` would raise
a bare ``TypeError`` from the comparison itself instead of a clear message
naming the field. This closes that hole the same way ``catalog.py`` already
does for interval bounds and measurement values.

These settings are caller-constructed configuration, not measurement-pipeline
output, so unlike ``catalog.py``'s NumPy-scalar coercion, no NumPy handling is
added here: a caller building a ``np.float64`` threshold can call ``.item()``
itself, the same way any other non-native-Python value would need to.
"""


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
