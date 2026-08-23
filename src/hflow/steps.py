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
import logging
import math
import re
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum, StrEnum
from fnmatch import fnmatchcase
from pathlib import Path
from types import CodeType, ModuleType
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from hflow.episode import Episode
    from hflow.resample import DerivedSeries

# One scalar per key, and no None. A NumPy scalar is coerced to its Python
# equivalent at the catalog boundary; anything else is refused there, naming
# the check and key. A check with nothing to say omits the key rather than
# measuring None: a NULL value column means nothing stored a value, so
# admitting None would make "not measured" and "silently dropped" the same
# row (see #126).
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


class Comparison(StrEnum):
    """How a threshold compares a measurement to its value. Both inclusive."""

    AT_MOST = "at_most"  # accept while measurement <= value
    AT_LEAST = "at_least"  # accept while measurement >= value


class Aggregation(StrEnum):
    """How a threshold folds the several keys one pattern matches.

    Measurement keys carry run-time topic names, so one pattern routinely
    matches once per camera. ``EVERY_KEY`` reads "no camera may be blacked
    out"; ``ANY_KEY`` reads "at least one camera is usable".
    """

    EVERY_KEY = "every_key"
    ANY_KEY = "any_key"


@dataclass(frozen=True)
class Threshold:
    """One accept condition over the measurement keys a glob matches.

    ``key_pattern`` is an :func:`fnmatch.fnmatchcase` glob over measurement
    keys. ``*`` crosses ``/``, so ``*/black_frame_pct`` matches
    ``/wrist_cam/compressed/black_frame_pct`` but NOT a bare
    ``black_frame_pct``; write ``*black_frame_pct`` to cover both.
    """

    key_pattern: str
    comparison: Comparison
    value: float
    across: Aggregation = Aggregation.EVERY_KEY

    def __post_init__(self) -> None:
        if not self.key_pattern:
            raise ValueError(
                "threshold key_pattern must not be empty; it is a glob over "
                "measurement keys, e.g. '*/black_frame_pct'"
            )
        if math.isnan(self.value):
            raise ValueError(
                f"threshold value for {self.key_pattern!r} must be a number, not NaN: "
                "every comparison against NaN is False, so this gate would reject "
                "every episode it evaluated"
            )

    def holds(self, measurement: float) -> bool:
        match self.comparison:
            case Comparison.AT_MOST:
                return measurement <= self.value
            case Comparison.AT_LEAST:
                return measurement >= self.value


@dataclass(frozen=True)
class Gate:
    """A verdict policy a pipeline attaches to one check: ``@app.check(gate=...)``.

    Every threshold must hold for the episode to be accepted. HFlow ships
    recommended gates as values (see ``hflow.checks``) and nothing gates until
    a pipeline passes one in -- so a threshold can have a documented default
    without a corpus ever being quarantined by a number its owner never chose.

    The runner evaluates a gate over the evidence the check returned, never
    inside the check: a threshold applied inside would raise on a key that
    episode never produced, and the runner would record that as an
    infrastructure error, discarding every measurement already computed.

    A gate reads only the measurements of the check it is attached to. Policy
    spanning checks is a curation query (docs/ARCHITECTURE.md layer 3): a
    cross-check gate would depend on registration order and its version would
    lie about what it depends on.
    """

    accept_when: tuple[Threshold, ...]

    def __post_init__(self) -> None:
        if not self.accept_when:
            raise ValueError(
                "Gate(accept_when=()) holds no thresholds, so it would accept every "
                "episode without reading a measurement -- a verdict claiming a check "
                "passed when nothing was checked. Pass at least one threshold, e.g.\n\n"
                "    hflow.Gate(accept_when=(\n"
                '        hflow.Threshold("*/black_frame_pct", hflow.Comparison.AT_MOST, 50.0),\n'
                "    ))\n"
            )


@dataclass(frozen=True)
class GateDecided:
    """The gate owns this verdict."""

    verdict: bool


@dataclass(frozen=True)
class GateAbstained:
    """The gate cannot claim a pass, so it offers no verdict.

    The named patterns matched no measurement key at all (a key this episode
    never produced, or a typo), or matched values that are not real numbers.
    Silence is the honest outcome: a partial conjunction reported as a pass
    would be a quality claim about evidence nobody looked at.
    """

    unevaluated_patterns: tuple[str, ...]


GateDecision = GateDecided | GateAbstained

# Recorded when a gate could not read a threshold, so one aimed at a key that
# never exists is a query away instead of invisible.
GATE_UNEVALUATED_TAG_PREFIX = "gate-unevaluated:"


def evaluate_gate(gate: Gate, measurements: Mapping[str, MeasurementValue]) -> GateDecision:
    """Fold a gate over one check's measurements. Pure: no episode, no I/O.

    A gate is a conjunction, and the two outcomes are not symmetric. Rejecting
    needs only one threshold to fail -- no amount of unread evidence can make a
    conjunction true once a conjunct is false -- so a reject stands even when
    other thresholds could not be evaluated. Accepting needs every threshold
    read, so an unevaluable one abstains rather than passing: otherwise a
    typo'd key would quietly weaken a gate into approving what it never
    inspected.
    """
    unevaluated: list[str] = []
    verdict = True
    for threshold in gate.accept_when:
        matched = [
            value
            for key, value in sorted(measurements.items())
            if fnmatchcase(key, threshold.key_pattern)
        ]
        comparable = [
            float(value)
            for value in matched
            if isinstance(value, int | float)
            and not isinstance(value, bool)
            and not math.isnan(value)
        ]
        # Evaluable only when at least one key matched AND every match held a
        # real number. Ignoring the odd text or NaN value would evaluate a
        # partial conjunction and report it as a whole one.
        if not comparable or len(comparable) != len(matched):
            unevaluated.append(threshold.key_pattern)
            continue
        holds = [threshold.holds(value) for value in comparable]
        match threshold.across:
            case Aggregation.EVERY_KEY:
                verdict = verdict and all(holds)
            case Aggregation.ANY_KEY:
                verdict = verdict and any(holds)
    # A failed conjunct settles the conjunction, so reject even while blind to
    # the rest; only claiming a pass requires having read everything.
    if not verdict:
        return GateDecided(verdict=False)
    if unevaluated:
        return GateAbstained(unevaluated_patterns=tuple(unevaluated))
    return GateDecided(verdict=True)


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
    gate: Gate | None = None


def compute_check_version(
    name: str,
    function: Callable[..., object],
    critical: bool,
    requires: frozenset[str],
    uses: str | None,
    declared_version: str | None = None,
    *,
    gate: "Gate | None" = None,
) -> str:
    """Content-hash a step, or record the version its author declared.

    Used for checks and enrichments alike (enrichments pass
    ``critical=False``). By default the version is DERIVED: the function's
    source, its closure values and defaults, and -- transitively, across
    first-party modules only (see :class:`_IdentityScope`) -- the helpers it
    calls and the constants they read. A parser or a threshold one call below
    the step still moves the step's version.

    Pass ``declared_version`` to own the version instead. Nothing derived from
    the function is hashed then, so a refactor the author judges equivalent
    keeps the identity and its rows stay comparable; the cost is that they
    must remember to bump it when behavior really does change. Use it for an
    opaque client the machinery cannot read, and for a step stable enough that
    its author would rather promise than measure.

    Deriving is the default because the two mistakes are not symmetric. An
    unnecessary version split is recoverable -- both hashes exist and a query
    can union them. A missed one is not: two behaviors under one version, with
    nothing left to say which row came from which.

    A ``gate`` arrives from registration rather than from the function, so it
    is folded in under BOTH modes: tuning a threshold has to move the version,
    or two policies share one and curation can no longer pin either.
    """
    identity = json.dumps(
        step_identity_payload(
            name, function, critical, requires, uses, declared_version, gate=gate
        ),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(identity.encode()).hexdigest()[:12]


def step_identity_payload(
    name: str,
    function: Callable[..., object],
    critical: bool,
    requires: frozenset[str],
    uses: str | None,
    declared_version: str | None = None,
    *,
    gate: "Gate | None" = None,
) -> dict[str, VersionIdentityValue]:
    """Everything :func:`compute_check_version` hashes, before hashing it.

    Separate from the hash so the description can be inspected rather than
    only compared. A hash answers "did this change"; only the payload answers
    "what does this version actually cover", which is what
    ``UNDESCRIBED_CONFIGURATION_KEY`` makes checkable -- a gap in the walk is
    otherwise invisible until someone edits the code it failed to read.

    Two identity modes, and ``declared_version`` chooses between them:

    - **Derived** (the default): the implementation and everything it
      transitively reaches decide the version. Nothing to remember, and the
      version cannot claim two behaviors are one.
    - **Declared**: the author owns the version outright and NOTHING derived
      from the function reaches the hash, so a refactor they judge equivalent
      keeps the identity and its rows stay comparable.

    What survives declaring is the REGISTRATION: name, ``critical``,
    ``requires``, ``uses``, and the gate. Those are not what the function is
    made of, they are what the author wrote at the decorator -- changing one
    is a deliberate edit with a visible diff, not a refactor, and a gate in
    particular must keep moving the version or two threshold policies share
    one and curation can pin neither (see :class:`Gate`).
    """
    if declared_version is not None and not declared_version:
        raise ValueError(f"declared step version must not be empty for step {name!r}")

    scope = _identity_scope_for(function)
    identity: dict[str, VersionIdentityValue] = {
        "name": name,
        "critical": critical,
        "requires": sorted(requires),
        "uses": uses,
        # Present only when a gate is declared, like transform.py's
        # resample_policy: an unconditional key would re-version every
        # check, enrichment, and derived channel -- and derived-channel
        # versions reach pipeline_version, which is stamped inside the
        # bytes episode_id hashes. One new key would restamp every corpus.
        **({"gate": _stable_version_identity_value(gate, scope)} if gate is not None else {}),
    }
    if declared_version is not None:
        # Not introspected at all, rather than introspected and overridden:
        # an opaque vendor client has no describable implementation, and an
        # ordinary function whose author declared a version must not have its
        # source leak back in -- that would make the declaration advisory and
        # the refactor it exists to permit would re-version anyway.
        identity["declared_version"] = declared_version
        return identity
    identity["implementation"] = _callable_implementation_identity(function)
    identity["configuration"] = _callable_behavior_configuration(function, scope=scope)
    return identity


# Marks a point where the identity walk could not describe a transitively
# reached helper's own state and recorded its NAME instead. Named so the gap
# is greppable in a payload rather than silent: hflow's own built-ins must
# never produce one (tests/test_processing_regressions.py pins that), because
# a marker here means editing the code below it would not move a version.
UNDESCRIBED_CONFIGURATION_KEY = "undescribed_configuration_of"


@dataclass(frozen=True)
class _IdentityScope:
    """How far a step's identity follows the code that step calls.

    A step hash covers the source of every function the step NAMES, but until
    this scope existed it stopped there: editing a constant or a parser one
    call deeper changed what the step measured while its version stood still,
    and the new rows appended under the old version -- two behaviors sharing
    one identity, which is the failure this whole module exists to prevent.
    The built-ins made that concrete, ``camera_signal_quality`` naming
    ``frame_stats`` while the instrument's own parsing sat a level below.

    ``first_party_roots`` bounds what "deeper" may mean: hflow's own code,
    plus the top-level package the step itself is defined in (so a pipeline's
    private helpers count, wherever the author put them). A DEPENDENCY is
    deliberately not followed -- folding numpy's source into a step version
    would re-version a corpus on an unrelated numpy release, which is exactly
    the release-number coupling :mod:`hflow.behavior` removed.

    ``visited`` terminates the walk on recursion, direct or mutual, and keeps
    a diamond from being described twice.
    """

    first_party_roots: frozenset[str]
    visited: frozenset[str] = frozenset()

    def follows(self, function: object) -> bool:
        module_name = getattr(function, "__module__", None)
        if not isinstance(module_name, str) or not module_name:
            return False
        return module_name.partition(".")[0] in self.first_party_roots

    def entered(self, identity_key: str) -> "_IdentityScope":
        return _IdentityScope(self.first_party_roots, self.visited | {identity_key})


def _identity_key(function: object) -> str:
    """A stable name for one function, for cycle detection and for naming it
    in the hash when its configuration cannot be described."""
    module_name = getattr(function, "__module__", None) or "?"
    qualified_name = getattr(function, "__qualname__", None) or "?"
    return f"{module_name}.{qualified_name}"


def _identity_scope_for(function: Callable[..., object]) -> _IdentityScope:
    """The first-party roots for one step: hflow, plus the step's own package."""
    target: object = function.func if isinstance(function, functools.partial) else function
    module_name = getattr(target, "__module__", None)
    roots = {"hflow"}
    if isinstance(module_name, str) and module_name:
        roots.add(module_name.partition(".")[0])
    return _IdentityScope(first_party_roots=frozenset(roots))


def _callable_implementation_identity(
    function: Callable[..., object],
) -> VersionIdentityValue:
    """What a callable IS, or a refusal naming the way out.

    Only reached for a step whose version is derived, so there is no "describe
    it loosely" mode: a callable this cannot read has no honest derived
    identity, and the refusal points at ``version='...'`` -- which now means
    the function is not introspected at all rather than introspected weakly.
    """
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
    *,
    scope: _IdentityScope | None = None,
) -> VersionIdentityValue:
    configuration: dict[str, VersionIdentityValue] = {}
    if isinstance(function, functools.partial):
        configuration["partial_args"] = _stable_version_identity_value(function.args, scope)
        configuration["partial_keywords"] = _stable_version_identity_value(function.keywords, scope)
    elif inspect.isfunction(function) or inspect.ismethod(function):
        closure_variables = inspect.getclosurevars(function)
        configuration["nonlocals"] = _stable_version_identity_value(
            closure_variables.nonlocals, scope
        )
        configuration["globals"] = _stable_version_identity_value(closure_variables.globals, scope)
        configuration["defaults"] = _stable_version_identity_value(
            getattr(function, "__defaults__", None), scope
        )
        configuration["keyword_defaults"] = _stable_version_identity_value(
            getattr(function, "__kwdefaults__", None), scope
        )
    else:
        try:
            callable_state = vars(function)
        except TypeError:
            callable_state = {}
        configuration["callable_state"] = _stable_version_identity_value(callable_state, scope)
    return configuration


def _stable_version_identity_value(
    value: object, scope: _IdentityScope | None = None
) -> VersionIdentityValue:
    """Convert configuration to deterministic JSON or refuse opaque state.

    ``scope`` carries the first-party traversal; without one this describes a
    referenced function by its source alone, which is what callers outside
    step versioning want.
    """
    # Unwrap decorators before deciding what a value is. A memoised helper is
    # its wrapped function plus a cache, and the cache changes no behavior a
    # step can observe -- so hashing the function underneath is both correct
    # and the difference between versioning a step and refusing to register
    # it (hflow.ffmpeg._binary memoises the binary probes the instrument
    # calls). inspect.unwrap raises on a __wrapped__ cycle; keep the wrapper.
    if not isinstance(value, type) and callable(value) and hasattr(value, "__wrapped__"):
        with suppress(ValueError):
            value = inspect.unwrap(value)
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
    if isinstance(value, re.Pattern):
        # A compiled pattern IS behavior, and a common way to express it: the
        # ffmpeg instrument parses its output with two module-level patterns,
        # so editing one changes what every camera check measures. Described
        # by the pattern and its flags -- the compiled object is opaque, but
        # what it was compiled FROM is exactly the thing that can change.
        return {"regex": value.pattern, "regex_flags": int(value.flags)}
    if isinstance(value, logging.Logger):
        # A logger is infrastructure a step carries, never behavior a step
        # has. Named rather than refused, so a module that logs stays
        # describable; its handlers and level are runtime state and must NOT
        # reach a version, or the same code would hash differently under a
        # different logging configuration.
        return {"logger": value.name}
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
        referenced_key = _identity_key(value)
        identity: dict[str, VersionIdentityValue] = {
            "callable": referenced_key,
            "implementation": _callable_implementation_identity(value),
        }
        if scope is not None and referenced_key not in scope.visited and scope.follows(value):
            # Follow first-party code one call further: this function's own
            # constants, defaults and callees are part of what the step does.
            # A value the walk cannot describe is NAMED here rather than
            # raised on -- refusing would break registration of a step over
            # some helper's private state, while the step's OWN captured
            # state stays strict because that refusal is raised outside this
            # branch (see compute_check_version). The name still moves the
            # hash if the helper is swapped for a different one.
            try:
                identity["configuration"] = _callable_behavior_configuration(
                    value, scope=scope.entered(referenced_key)
                )
            except (ValueError, TypeError):
                identity["configuration"] = {UNDESCRIBED_CONFIGURATION_KEY: referenced_key}
        return identity
    if inspect.isbuiltin(value):
        return {"builtin": f"{value.__module__}.{value.__qualname__}"}
    if isinstance(value, list | tuple):
        return [_stable_version_identity_value(item, scope) for item in value]
    if isinstance(value, set | frozenset):
        encoded_items = [_stable_version_identity_value(item, scope) for item in value]
        return sorted(
            encoded_items,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    if isinstance(value, dict):
        encoded_entries = [
            {
                "key": _stable_version_identity_value(key, scope),
                "value": _stable_version_identity_value(item, scope),
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
                    getattr(value, dataclass_field.name), scope
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
