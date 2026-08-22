"""Step types: what a quality check receives and returns.

A check is a plain function ``(Episode) -> CheckResult``. It records
*evidence* -- measurements, time intervals, tags -- not verdicts; pass/fail
policy belongs to the consumer at curation time (docs/ARCHITECTURE.md,
"Quality checks and curation"). A check MAY declare a user-owned ``verdict``:
on a check registered with ``critical=True``, a False verdict quarantines the
episode (a tag, never a deletion) and skips its downstream steps.
"""

import functools
import hashlib
import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum, StrEnum
from pathlib import Path
from types import CodeType, ModuleType
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from hflow.episode import Episode
    from hflow.resample import DerivedSeries

MeasurementValue = float | int | str | bool
VersionIdentityValue: TypeAlias = (
    bool
    | int
    | float
    | str
    | list["VersionIdentityValue"]
    | dict[str, "VersionIdentityValue"]
    | None
)


class Stage(StrEnum):
    """The ingest stage graph's toggleable sub-DAGs, as stage names shared with the DAGs.

    These strings are conf vocabulary: the master DAG resolves a run profile
    to a stage set and triggers only the sub-DAGs it names, and
    ``App.process(stages=...)`` runs the same set in-process. One owner --
    here -- so the runner and the DAG bundle can never disagree.
    """

    SYNC = "sync"  # "Transform & sync": the canonical transform (critical path)
    META = "meta"  # "Metadata": checks + catalog registration
    LABELS = "labels"  # "Labels & artifacts": enrichments (non-critical)
    MEDIA = "media"  # "Media": derived media artifacts (contact sheets)


class IngestMode(StrEnum):
    """The trigger conf's ``mode`` vocabulary, shared with the DAGs.

    Like :class:`Stage`, these strings cross the conf boundary: the master
    DAG validates against a copy baked at render time, and
    ``hflow.stage_execution`` parses the value at the library boundary. One
    owner -- here -- so the runner and the DAGs can never disagree.
    """

    BATCH = "batch"  # bin-packed near-equal-byte shards with staggered starts
    ONLINE = "online"  # latency-first: one immediate batch, no stagger


# Run profiles: the same stage graph with different sub-DAGs enabled.
RUN_PROFILES: dict[str, frozenset[Stage]] = {
    "full": frozenset(Stage),
    "metadata_backfill": frozenset({Stage.META}),
    "relabel": frozenset({Stage.LABELS}),
}


def stages_for_profile(name: str) -> frozenset[Stage]:
    """Resolve a run-profile name to its enabled stage set, loudly."""
    try:
        return RUN_PROFILES[name]
    except KeyError:
        raise ValueError(
            f"unknown run profile {name!r}; valid profiles: {sorted(RUN_PROFILES)}"
        ) from None


class CheckStatus(StrEnum):
    """Outcome classification of one step invocation.

    Lives here (not in ``app``) because it is stored vocabulary: the catalog
    records it and curation filters on it, and both must share the one
    definition with the runner.
    """

    PASSED = "passed"  # verdict True
    FAILED = "failed"  # verdict False (quarantines the episode when critical)
    MEASURED = "measured"  # ran and recorded evidence; no verdict offered
    SKIPPED = "skipped"  # not run (episode quarantined upstream)
    ERROR = "error"  # crashed: infrastructure, not data


@dataclass(frozen=True)
class Interval:
    """A labeled time span inside an episode, in nanoseconds of log time."""

    start_ns: int
    end_ns: int
    label: str = ""


@dataclass
class CheckResult:
    """What a check returns. Everything here lands in the episode's catalog row.

    :param measurements: Named numeric/string facts (catalog columns).
    :param intervals: Labeled time spans (e.g. detected freezes).
    :param tags: Free-form labels routed to the catalog.
    :param verdict: Optional user-owned pass/fail. ``None`` means the check
        offers evidence only. A False verdict on a ``critical`` check
        quarantines the episode.
    """

    measurements: dict[str, MeasurementValue] = field(default_factory=dict)
    intervals: list[Interval] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    verdict: bool | None = None


CheckFunction = Callable[["Episode"], CheckResult]


@dataclass
class EnrichmentResult:
    """What an enrichment returns: derived labels and artifacts, no verdicts.

    Enrichments are the "Labels & artifacts" stage family: performance
    labels, video captions, segmentations (the examples named in Dyna's
    article). They never gate anything -- gate semantics belong to critical
    checks -- and they never run on a quarantined episode (no enrichment
    spend on an episode with a dead camera).

    :param labels: Derived facts (captions, scores); recorded exactly like
        check measurements, so they are catalog columns at curation time.
    :param artifacts: Named files the enrichment wrote (masks, sheets);
        recorded as text measurements under ``artifact/<name>`` holding the
        published file's URI. The runner copies or uploads external files
        into the episode's artifact directory; ``ep.workdir`` remains scratch.
    :param tags: Free-form labels routed to the catalog.
    """

    labels: dict[str, MeasurementValue] = field(default_factory=dict)
    artifacts: dict[str, Path] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


EnrichmentFunction = Callable[["Episode"], EnrichmentResult]


DerivedFunction = Callable[["Episode"], "DerivedSeries"]


@dataclass(frozen=True)
class DerivedChannel:
    """A derived-signal function plus its output topic and identity.

    The function is computed over the SOURCE episode during the transform and
    its :class:`~hflow.resample.DerivedSeries` is written as a new channel
    on ``topic``. ``version`` is the same content hash checks use
    (:func:`compute_check_version`), stamped into ``provenance/v1`` so a
    changed derivation is a new measurement identity, never a silent rewrite.
    """

    topic: str
    function: DerivedFunction
    version: str


@dataclass(frozen=True)
class RegisteredEnrichment:
    """An enrichment function plus its registration configuration."""

    name: str
    function: EnrichmentFunction
    requires: frozenset[str]
    uses: str | None
    version: str


@dataclass(frozen=True)
class RegisteredCheck:
    """A check function plus its registration configuration.

    ``version`` is a content hash of the check's configuration and source, so
    re-running a changed check appends new-version measurement rows instead of
    silently overwriting (the mixed-version-corpus rule applied to checks).
    """

    name: str
    function: CheckFunction
    critical: bool
    requires: frozenset[str]
    uses: str | None
    version: str


def compute_check_version(
    name: str,
    function: Callable[..., object],
    critical: bool,
    requires: frozenset[str],
    uses: str | None,
    declared_version: str | None = None,
) -> str:
    """Content-hash a step's implementation and behavior configuration.

    Used for checks and enrichments alike (enrichments pass
    ``critical=False``). Closure values, defaults, and callable-instance state
    are included when they can be represented deterministically. Pass an
    explicit ``declared_version`` for opaque clients or other configuration
    that cannot be inspected safely.
    """
    if declared_version is not None and not declared_version:
        raise ValueError(f"declared step version must not be empty for step {name!r}")

    implementation_identity = _callable_implementation_identity(
        function,
        allow_opaque=declared_version is not None,
    )
    behavior_configuration: VersionIdentityValue
    if declared_version is not None:
        behavior_configuration = {"declared_version": declared_version}
    else:
        behavior_configuration = _callable_behavior_configuration(function)

    identity = json.dumps(
        {
            "name": name,
            "critical": critical,
            "requires": sorted(requires),
            "uses": uses,
            "implementation": implementation_identity,
            "configuration": behavior_configuration,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(identity.encode()).hexdigest()[:12]


def _callable_implementation_identity(
    function: Callable[..., object],
    *,
    allow_opaque: bool = False,
) -> VersionIdentityValue:
    if isinstance(function, functools.partial):
        return {
            "partial_func": _callable_implementation_identity(function.func),
            "partial_args": _stable_version_identity_value(function.args),
            "partial_keywords": _stable_version_identity_value(function.keywords),
        }
    source_target: object = function
    if not inspect.isfunction(function) and not inspect.ismethod(function):
        source_target = type(function).__call__
    try:
        return {"source": inspect.getsource(source_target)}
    except (OSError, TypeError):
        code = getattr(source_target, "__code__", None)
        if isinstance(code, CodeType):
            return {"code": _code_identity(code)}
    if allow_opaque:
        return {
            "opaque_callable_type": f"{type(function).__module__}.{type(function).__qualname__}"
        }
    raise ValueError(
        f"cannot derive a stable implementation identity for {function!r}; "
        "register it with version='...'"
    )


def _code_identity(code: CodeType) -> VersionIdentityValue:
    constants: list[VersionIdentityValue] = []
    for constant in code.co_consts:
        if isinstance(constant, CodeType):
            constants.append({"nested_code": _code_identity(constant)})
        else:
            constants.append(_stable_version_identity_value(constant))
    return {
        "bytecode": code.co_code.hex(),
        "constants": constants,
        "names": list(code.co_names),
        "variable_names": list(code.co_varnames),
        "free_variables": list(code.co_freevars),
        "cell_variables": list(code.co_cellvars),
    }


def _callable_behavior_configuration(
    function: Callable[..., object],
) -> VersionIdentityValue:
    configuration: dict[str, VersionIdentityValue] = {}
    if isinstance(function, functools.partial):
        configuration["partial_args"] = _stable_version_identity_value(function.args)
        configuration["partial_keywords"] = _stable_version_identity_value(function.keywords)
    elif inspect.isfunction(function) or inspect.ismethod(function):
        closure_variables = inspect.getclosurevars(function)
        configuration["nonlocals"] = _stable_version_identity_value(closure_variables.nonlocals)
        configuration["globals"] = _stable_version_identity_value(closure_variables.globals)
        configuration["defaults"] = _stable_version_identity_value(
            getattr(function, "__defaults__", None)
        )
        configuration["keyword_defaults"] = _stable_version_identity_value(
            getattr(function, "__kwdefaults__", None)
        )
    else:
        try:
            callable_state = vars(function)
        except TypeError:
            callable_state = {}
        configuration["callable_state"] = _stable_version_identity_value(callable_state)
    return configuration


def _stable_version_identity_value(value: object) -> VersionIdentityValue:
    """Convert configuration to deterministic JSON or refuse opaque state."""
    if isinstance(value, Enum):
        return {
            "enum_type": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": _stable_version_identity_value(value.value),
        }
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, Path):
        return {"path": str(value)}
    if isinstance(value, ModuleType):
        # The module's IDENTITY, never its ``__version__``: a version number
        # is a poor proxy for "does this library compute differently", and
        # folding it in re-versioned every step that merely referenced
        # ``hflow`` or ``numpy`` on any release of those packages -- including
        # releases that changed nothing a step can observe. What a step
        # actually does still lives in its own source and captured values,
        # which this hash covers directly.
        return {"module": value.__name__}
    if isinstance(value, type):
        return {"type": f"{value.__module__}.{value.__qualname__}"}
    if isinstance(value, CodeType):
        return {"code": _code_identity(value)}
    if inspect.isfunction(value) or inspect.ismethod(value):
        return {
            "callable": f"{value.__module__}.{value.__qualname__}",
            "implementation": _callable_implementation_identity(value),
        }
    if inspect.isbuiltin(value):
        return {"builtin": f"{value.__module__}.{value.__qualname__}"}
    if isinstance(value, list | tuple):
        return [_stable_version_identity_value(item) for item in value]
    if isinstance(value, set | frozenset):
        encoded_items = [_stable_version_identity_value(item) for item in value]
        return sorted(
            encoded_items,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    if isinstance(value, dict):
        encoded_entries = [
            {
                "key": _stable_version_identity_value(key),
                "value": _stable_version_identity_value(item),
            }
            for key, item in value.items()
        ]
        return {
            "mapping": sorted(
                encoded_entries,
                key=lambda entry: json.dumps(entry["key"], sort_keys=True),
            )
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "dataclass_type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                dataclass_field.name: _stable_version_identity_value(
                    getattr(value, dataclass_field.name)
                )
                for dataclass_field in fields(value)
            },
        }
    if value is Ellipsis:
        return {"singleton": "Ellipsis"}
    raise ValueError(
        f"cannot derive a stable version identity for captured value {value!r} "
        f"({type(value).__module__}.{type(value).__qualname__}); register the step "
        "with version='...'"
    )
