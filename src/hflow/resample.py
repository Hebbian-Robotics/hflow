"""Grid resampling for derived signals: pure helpers over :class:`ChannelData`.

Multi-rate alignment is where format converters silently diverge
(docs/ARCHITECTURE.md, "Transform"), so the policy here is explicit and
versioned: ``hflow.format.RESAMPLE_POLICY_VERSION`` names the semantics
below and is stamped into ``provenance/v1`` whenever derived channels are
written.

Resample policy version "2":

- The grid steps by ``1e9 / rate_hz`` nanoseconds and is anchored to integer
  multiples of the step (epoch-anchored): it runs from the first multiple at
  or after the channel's first timestamp to the last multiple at or before
  its last timestamp. Epoch anchoring makes grids from different channels at
  the same rate share timestamps, so derived signals align without a second
  resampling pass.
- ``interpolate``: linear interpolation per column (``numpy.interp``).
- ``nearest``: the value of the source sample nearest in time to each grid
  point; the earlier sample wins an exact tie.
- ``slerp``: shortest-arc spherical interpolation of four-component rotation
  fields after source normalization and sequential hemisphere alignment.
"""

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Literal, assert_never

import numpy as np

from hflow.format import NANOSECONDS_PER_SECOND

if TYPE_CHECKING:
    from hflow.episode import ChannelData

ResamplePolicy = Literal["interpolate", "nearest", "slerp"]

_SLERP_NEAR_PARALLEL_DOT = 1.0 - 1e-7


@dataclass(frozen=True)
class DerivedSeries:
    """A derived signal on a timestamp grid: what a derive function returns.

    :param timestamps_ns: Grid timestamps in nanoseconds (int64, shape (n,)).
    :param values: Column name -> per-grid-point values (each shape (n,)).
    """

    timestamps_ns: np.ndarray
    values: dict[str, np.ndarray]

    def __post_init__(self) -> None:
        # Parse at the boundary: user derive functions construct these, so
        # normalize to arrays and fail loudly on shape mismatches here rather
        # than deep inside the transform's write loop.
        timestamps = np.asarray(self.timestamps_ns, dtype=np.int64)
        if timestamps.ndim != 1:
            raise ValueError(f"timestamps_ns must be one-dimensional, got shape {timestamps.shape}")
        normalized_values: dict[str, np.ndarray] = {}
        for column_name, column_values in self.values.items():
            column_array = np.asarray(column_values)
            if column_array.ndim != 1:
                raise ValueError(
                    f"derived column {column_name!r} must be one-dimensional "
                    f"(one scalar per grid point), got shape {column_array.shape}"
                )
            if len(column_array) != len(timestamps):
                raise ValueError(
                    f"derived column {column_name!r} has {len(column_array)} values "
                    f"but the grid has {len(timestamps)} timestamps"
                )
            normalized_values[column_name] = column_array
        object.__setattr__(self, "timestamps_ns", timestamps)
        object.__setattr__(self, "values", normalized_values)

    def __len__(self) -> int:
        return len(self.timestamps_ns)


def _grid_timestamps_ns(
    first_timestamp_ns: int, last_timestamp_ns: int, rate_hz: float
) -> np.ndarray:
    """Epoch-anchored grid timestamps covering [first, last] at ``rate_hz``.

    Exact ``Fraction`` arithmetic: at epoch-scale nanosecond timestamps
    (~1.7e18) float64 only resolves ~256 ns, which would smear grid points.
    """
    step_ns = Fraction(NANOSECONDS_PER_SECOND) / Fraction(rate_hz)
    first_grid_index = math.ceil(Fraction(first_timestamp_ns) / step_ns)
    last_grid_index = math.floor(Fraction(last_timestamp_ns) / step_ns)
    if last_grid_index < first_grid_index:
        raise ValueError(
            f"no {rate_hz:g} Hz grid point falls inside "
            f"[{first_timestamp_ns}, {last_timestamp_ns}] ns; the channel span is "
            "shorter than one grid step -- lower the rate or use a longer channel"
        )
    return np.asarray(
        [
            round(grid_index * step_ns)
            for grid_index in range(first_grid_index, last_grid_index + 1)
        ],
        dtype=np.int64,
    )


def _source_columns(channel: "ChannelData", field: str | None) -> dict[str, np.ndarray]:
    """One named 1-D column per component of the channel's numeric field."""
    source_values = channel.to_numpy(field=field)
    # ``to_numpy(field=None)`` picks the field itself and does not report the
    # chosen name, so columns fall back to the neutral base name "value";
    # pass ``field=`` to name columns after the field.
    column_base_name = field if field is not None else "value"
    if source_values.ndim == 1:
        return {column_base_name: source_values}
    if source_values.ndim == 2:
        return {
            f"{column_base_name}_{component_index}": source_values[:, component_index]
            for component_index in range(source_values.shape[1])
        }
    raise ValueError(
        f"field {field!r} of topic {channel.topic!r} has {source_values.ndim} dimensions "
        "per message; to_grid supports scalar and flat-vector fields"
    )


# Componentwise interpolation of quaternions produces unnormalized rotations
# and is wrong across antipodal sign flips. Name-based detection is best-effort:
# it cannot catch every quaternion field, but it catches the conventional ones
# (orientation/quaternion/rotation, 4 columns).
_QUATERNION_NAME_HINTS = ("quat", "orientation", "rotation")


def _looks_like_quaternion(field: str | None, schema_name: str, column_count: int) -> bool:
    if column_count != 4:
        return False
    lowered_field = (field or "").lower()
    if any(hint in lowered_field for hint in _QUATERNION_NAME_HINTS):
        return True
    return "quaternion" in schema_name.lower()


def _nearest_source_indices(
    source_timestamps_ns: np.ndarray, grid_timestamps_ns: np.ndarray
) -> np.ndarray:
    """Index of the source sample nearest each grid point (earlier wins ties).

    Exact int64 comparisons -- no float conversion of epoch timestamps.
    """
    last_source_index = len(source_timestamps_ns) - 1
    insertion_points = np.searchsorted(source_timestamps_ns, grid_timestamps_ns)
    earlier = np.clip(insertion_points - 1, 0, last_source_index)
    later = np.clip(insertion_points, 0, last_source_index)
    distance_to_earlier = grid_timestamps_ns - source_timestamps_ns[earlier]
    distance_to_later = source_timestamps_ns[later] - grid_timestamps_ns
    return np.where(distance_to_earlier <= distance_to_later, earlier, later)


def _slerp_quaternions(
    source_timestamps_ns: np.ndarray,
    source_quaternions: np.ndarray,
    grid_timestamps_ns: np.ndarray,
) -> np.ndarray:
    """Shortest-arc interpolation of four-component rotations.

    Component order is deliberately opaque: dot products, normalization, and
    scalar multiplication work identically for ``xyzw`` and ``wxyz`` input.
    """
    source_norms = np.linalg.norm(source_quaternions, axis=1)
    if np.any(~np.isfinite(source_quaternions)) or np.any(~np.isfinite(source_norms)):
        raise ValueError("slerp source quaternions must contain only finite values")
    if np.any(source_norms == 0.0):
        raise ValueError("slerp source quaternions must have nonzero norm")

    aligned = source_quaternions / source_norms[:, np.newaxis]
    aligned = aligned.copy()
    for source_index in range(1, len(aligned)):
        if np.dot(aligned[source_index - 1], aligned[source_index]) < 0.0:
            aligned[source_index] *= -1.0

    later_indices = np.searchsorted(source_timestamps_ns, grid_timestamps_ns, side="left")
    later_indices = np.clip(later_indices, 0, len(source_timestamps_ns) - 1)
    earlier_indices = np.maximum(later_indices - 1, 0)
    earlier_timestamps = source_timestamps_ns[earlier_indices]
    later_timestamps = source_timestamps_ns[later_indices]
    intervals = later_timestamps - earlier_timestamps
    interpolation_fractions = np.divide(
        grid_timestamps_ns - earlier_timestamps,
        intervals,
        out=np.zeros(len(grid_timestamps_ns), dtype=np.float64),
        where=intervals != 0,
    )

    earlier_quaternions = aligned[earlier_indices]
    later_quaternions = aligned[later_indices]
    pair_dots = np.clip(np.einsum("ij,ij->i", earlier_quaternions, later_quaternions), -1.0, 1.0)
    interpolated = np.empty_like(earlier_quaternions)

    # Below this angle, sin(theta) is small enough that the textbook slerp
    # quotient loses precision. 1e-7 follows the issue's requested scale and
    # keeps the normalized-lerp approximation far below float64 data noise.
    near_parallel = pair_dots > _SLERP_NEAR_PARALLEL_DOT
    if np.any(near_parallel):
        fractions = interpolation_fractions[near_parallel, np.newaxis]
        interpolated[near_parallel] = earlier_quaternions[near_parallel] + fractions * (
            later_quaternions[near_parallel] - earlier_quaternions[near_parallel]
        )

    spherical = ~near_parallel
    if np.any(spherical):
        theta = np.arccos(pair_dots[spherical])
        sin_theta = np.sin(theta)
        fractions = interpolation_fractions[spherical]
        earlier_weights = np.sin((1.0 - fractions) * theta) / sin_theta
        later_weights = np.sin(fractions * theta) / sin_theta
        interpolated[spherical] = (
            earlier_weights[:, np.newaxis] * earlier_quaternions[spherical]
            + later_weights[:, np.newaxis] * later_quaternions[spherical]
        )

    output_norms = np.linalg.norm(interpolated, axis=1)
    return interpolated / output_norms[:, np.newaxis]


def to_grid(
    channel: "ChannelData",
    rate_hz: float,
    *,
    policy: ResamplePolicy,
    field: str | None = None,
) -> DerivedSeries:
    """Resample one numeric channel field onto an epoch-anchored grid.

    Numeric fields only (reuses :meth:`ChannelData.to_numpy`, including its
    automatic field selection when ``field`` is None). A 2-D field of shape
    (n_messages, m) becomes m columns named ``<field>_0 .. <field>_{m-1}``.
    ``policy="slerp"`` requires exactly four columns but deliberately does not
    assume whether their source order is ``xyzw`` or ``wxyz``. Semantics are
    pinned by ``hflow.format.RESAMPLE_POLICY_VERSION`` (see the module
    docstring).
    """
    if rate_hz <= 0.0:
        raise ValueError(f"rate_hz must be positive, got {rate_hz}")
    source_columns = _source_columns(channel, field)
    source_timestamps_ns = channel.timestamps
    grid_timestamps_ns = _grid_timestamps_ns(
        int(source_timestamps_ns[0]), int(source_timestamps_ns[-1]), rate_hz
    )
    if policy == "interpolate" and _looks_like_quaternion(
        field, channel.info.schema_name, len(source_columns)
    ):
        raise ValueError(
            f"field {field!r} of topic {channel.topic!r} looks like a quaternion; "
            "componentwise interpolation corrupts rotations (unnormalized, wrong "
            "across sign flips). Use policy='slerp' for rotation-aware interpolation "
            "or policy='nearest' to copy source samples."
        )
    if policy == "slerp" and len(source_columns) != 4:
        raise ValueError(
            f"policy='slerp' requires a field with exactly 4 columns, got "
            f"{len(source_columns)} for field {field!r} of topic {channel.topic!r}"
        )
    match policy:
        case "interpolate":
            # Interpolation times are offsets from the first sample: relative
            # times fit float64 exactly where absolute epoch times do not.
            first_timestamp_ns = source_timestamps_ns[0]
            relative_source_times = (source_timestamps_ns - first_timestamp_ns).astype(np.float64)
            relative_grid_times = (grid_timestamps_ns - first_timestamp_ns).astype(np.float64)
            resampled_columns = {
                column_name: np.interp(
                    relative_grid_times, relative_source_times, column_values.astype(np.float64)
                )
                for column_name, column_values in source_columns.items()
            }
        case "nearest":
            nearest_indices = _nearest_source_indices(source_timestamps_ns, grid_timestamps_ns)
            resampled_columns = {
                column_name: column_values[nearest_indices]
                for column_name, column_values in source_columns.items()
            }
        case "slerp":
            source_quaternions = np.column_stack(list(source_columns.values())).astype(np.float64)
            resampled_quaternions = _slerp_quaternions(
                source_timestamps_ns, source_quaternions, grid_timestamps_ns
            )
            resampled_columns = {
                column_name: resampled_quaternions[:, component_index]
                for component_index, column_name in enumerate(source_columns)
            }
        case _:
            assert_never(policy)
    return DerivedSeries(timestamps_ns=grid_timestamps_ns, values=resampled_columns)
