"""Explicit, versioned projections of check results for external consumers.

Only selected scalar measurements and outcome status cross this boundary.
Paths, registrations, diagnostics, tags, and internal check identities do not.
"""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, TypedDict, assert_never

from hflow.app import Errored, Measured, NotRun, SkippedByQuarantine, SupersededByPipeline
from hflow.steps import MeasurementValue

if TYPE_CHECKING:
    from hflow.app import ProcessReport


@dataclass(frozen=True)
class CheckSelection:
    """Map a private check and its measurement keys to public output names."""

    check_name: str
    public_name: str
    measurements: Mapping[str, str]

    def __post_init__(self) -> None:
        selected = dict(self.measurements)
        if any(
            not isinstance(value, str) or not value
            for value in (self.check_name, self.public_name, *selected.keys(), *selected.values())
        ):
            raise ValueError("result selection names must be nonempty strings")
        object.__setattr__(self, "measurements", MappingProxyType(selected))


@dataclass(frozen=True)
class MeasuredCheck:
    name: str
    measurements: Mapping[str, MeasurementValue]
    verdict: bool | None

    def __post_init__(self) -> None:
        selected = dict(self.measurements)
        if (
            not isinstance(self.name, str)
            or not self.name
            or (self.verdict is not None and type(self.verdict) is not bool)
        ):
            raise ValueError("invalid measured check")
        for name, value in selected.items():
            if (
                not isinstance(name, str)
                or not name
                or type(value) not in (str, bool, int, float)
                or (isinstance(value, float) and not math.isfinite(value))
            ):
                raise ValueError("selected measurement is not a finite JSON scalar")
        object.__setattr__(self, "measurements", MappingProxyType(selected))


@dataclass(frozen=True)
class UnavailableCheck:
    name: str
    reason: Literal["error", "skipped", "superseded"]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or self.reason not in ("error", "skipped", "superseded")
        ):
            raise ValueError("invalid unavailable check")


class MeasuredCheckPayload(TypedDict):
    name: str
    status: Literal["measured"]
    measurements: dict[str, MeasurementValue]
    verdict: bool | None


class UnavailableCheckPayload(TypedDict):
    name: str
    status: Literal["error", "skipped", "superseded"]


class ResultProjectionPayload(TypedDict):
    schema_version: Literal[1]
    checks: list[MeasuredCheckPayload | UnavailableCheckPayload]


@dataclass(frozen=True)
class ResultProjection:
    checks: tuple[MeasuredCheck | UnavailableCheck, ...]

    def __post_init__(self) -> None:
        checks = tuple(self.checks)
        if len({check.name for check in checks}) != len(checks):
            raise ValueError("result projection names must be unique")
        object.__setattr__(self, "checks", checks)

    def to_payload(self) -> ResultProjectionPayload:
        checks: list[MeasuredCheckPayload | UnavailableCheckPayload] = []
        for check in self.checks:
            match check:
                case MeasuredCheck():
                    checks.append(
                        {
                            "name": check.name,
                            "status": "measured",
                            "measurements": dict(check.measurements),
                            "verdict": check.verdict,
                        }
                    )
                case UnavailableCheck():
                    checks.append({"name": check.name, "status": check.reason})
                case _ as unreachable:
                    assert_never(unreachable)
        return {"schema_version": 1, "checks": checks}

    def to_json(self, *, maximum_bytes: int) -> bytes:
        if type(maximum_bytes) is not int or maximum_bytes <= 0:
            raise ValueError("result byte limit must be positive")
        serialized = json.dumps(
            self.to_payload(), allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode()
        if len(serialized) > maximum_bytes:
            raise ValueError("result projection exceeded its byte limit")
        return serialized


def project_results(
    report: "ProcessReport", selections: tuple[CheckSelection, ...]
) -> ResultProjection:
    """Snapshot selected checks; unknown checks or missing measurements are errors.

    Check execution failures and deliberate skips retain distinct outcomes, but
    never include their diagnostic strings. A quality verdict remains separate
    from successful execution. There is no implicit export-all mode.
    """
    if len({selection.public_name for selection in selections}) != len(selections):
        raise ValueError("result projection names must be unique")
    checks: list[MeasuredCheck | UnavailableCheck] = []
    for selection in selections:
        # Inspect the canonical run state; do not infer failure from a nullable result.
        check_run = report.check(selection.check_name)
        match check_run.outcome:
            case Measured(result):
                measurements = {
                    public_name: result.measurements[measurement_name]
                    for public_name, measurement_name in selection.measurements.items()
                }
                checks.append(MeasuredCheck(selection.public_name, measurements, result.verdict))
            case Errored():
                checks.append(UnavailableCheck(selection.public_name, "error"))
            case NotRun(not_run):
                match not_run:
                    case SkippedByQuarantine():
                        checks.append(UnavailableCheck(selection.public_name, "skipped"))
                    case SupersededByPipeline():
                        checks.append(UnavailableCheck(selection.public_name, "superseded"))
                    case _ as unreachable_reason:
                        assert_never(unreachable_reason)
            case _ as unreachable:
                assert_never(unreachable)
    return ResultProjection(tuple(checks))
