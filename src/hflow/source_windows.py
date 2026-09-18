"""Complete source coverage using balanced, half-open millisecond windows.

Planning uses only a measured duration. It does not open media or assign work
to a scheduler, and can be imported without HFlow's media or pipeline stack.
"""

from dataclasses import dataclass

from hflow._field_guards import require_non_negative_int, require_positive_int


@dataclass(frozen=True)
class SourceWindow:
    """A nonempty interval ``[start_millis, end_millis)`` in source playback time."""

    start_millis: int
    end_millis: int

    def __post_init__(self) -> None:
        require_non_negative_int(self.start_millis, "start_millis")
        require_positive_int(self.end_millis, "end_millis")
        if self.end_millis <= self.start_millis:
            raise ValueError("end_millis must be greater than start_millis")

    @property
    def duration_millis(self) -> int:
        return self.end_millis - self.start_millis


def plan_source_windows(
    duration_millis: int,
    *,
    maximum_window_millis: int,
    minimum_window_millis: int = 1,
    maximum_windows: int = 10_000,
) -> tuple[SourceWindow, ...]:
    """Cover the entire source once, with no gaps, overlaps, or discarded tail.

    Use the fewest windows allowed by ``maximum_window_millis``, balancing
    their lengths to within one millisecond (longer windows first). Refuse an
    impossible minimum length or an excessive count before allocating results.
    Source identifiers, immutable revisions, and shard placement belong to
    the caller; the same measured duration and limits always yield this plan.
    """
    require_positive_int(duration_millis, "duration_millis")
    require_positive_int(maximum_window_millis, "maximum_window_millis")
    require_positive_int(minimum_window_millis, "minimum_window_millis")
    require_positive_int(maximum_windows, "maximum_windows")
    if minimum_window_millis > maximum_window_millis:
        raise ValueError("minimum_window_millis must not exceed maximum_window_millis")
    window_count = (duration_millis + maximum_window_millis - 1) // maximum_window_millis
    if window_count > maximum_windows:
        raise ValueError("complete source coverage exceeds maximum_windows")
    base_duration_millis, longer_window_count = divmod(duration_millis, window_count)
    if base_duration_millis < minimum_window_millis:
        raise ValueError("complete source coverage cannot satisfy minimum_window_millis")
    windows: list[SourceWindow] = []
    start_millis = 0
    for window_index in range(window_count):
        end_millis = start_millis + base_duration_millis + (window_index < longer_window_count)
        windows.append(SourceWindow(start_millis, end_millis))
        start_millis = end_millis
    return tuple(windows)
