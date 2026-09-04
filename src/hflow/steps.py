"""Step types: what a quality check receives and returns.

A check is a plain function ``(Episode) -> CheckResult``. It records
*evidence* -- measurements, timestamped observations, time intervals, tags -- not verdicts; pass/fail
policy belongs to the consumer at curation time (docs/ARCHITECTURE.md,
"Quality checks and curation"). A check MAY declare a user-owned ``verdict``:
on a check registered with ``critical=True``, a False verdict quarantines the
episode (a tag, never a deletion) and skips its downstream steps.
"""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from fnmatch import fnmatchcase
from pathlib import Path
from typing import TYPE_CHECKING, NewType

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

# A step version is an author-owned compatibility promise, not a digest of
# implementation details. Keeping it distinct after boundary parsing prevents
# internal code from accidentally substituting an arbitrary string.
StepVersion = NewType("StepVersion", str)


def parse_step_version(version: str) -> StepVersion:
    """Parse the explicit version shared by every registration surface."""
    if not isinstance(version, str):
        raise TypeError(f"step version must be a string, got {type(version).__name__}")
    if not version:
        raise ValueError("step version must not be empty")
    if version != version.strip():
        raise ValueError("step version must not have leading or trailing whitespace")
    return StepVersion(version)


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
    SKIPPED = "skipped"  # not run because the episode was quarantined upstream
    SUPERSEDED = "superseded"  # an auto-registered default the pipeline measures itself
    ERROR = "error"  # crashed: infrastructure, not data


# The statuses that mean "this check actually ran on this episode, and here is
# its evidence". One owner, because coverage denominators ask it and two
# private copies of "ran" is how two answers to one question drift apart.
#
# PASSED is deliberately not the whole set, and assuming it is yields an EMPTY
# answer: an evidence-only check offers no verdict, so it records MEASURED, and
# HFlow's entire built-in library is evidence-only by contract. FAILED counts
# too -- a check that ran and returned a False verdict measured the episode;
# whether to KEEP such an episode is a separate question its gate answers.
#
# ERROR is excluded, and that is a deliberate reading of
# docs/ARCHITECTURE.md's "a check crashing is infrastructure, not data": a
# crash means the evidence is missing, not that the episode is bad, so it
# lowers coverage and is a retry rather than a verdict about the recording.
RAN_STATUSES: tuple[CheckStatus, ...] = (
    CheckStatus.PASSED,
    CheckStatus.FAILED,
    CheckStatus.MEASURED,
)

# The statuses that mean "this step has had its turn on this exact episode, and
# running it again could not produce anything new". A different question from
# RAN_STATUSES, asked by the two places where the answer decides whether there
# is WORK LEFT TO DO rather than whether evidence exists: dataset membership
# (:func:`hflow.dataset.default_dataset_sql`) and stage planning
# (:mod:`hflow.stage_planning`).
#
# SUPERSEDED is the difference, and it is why this set has to exist separately.
# An auto-registered default that the pipeline's own step supersedes produces
# no evidence, so it is not a RAN status -- but it will stand down again on
# every episode forever, so there is no work in it. Wrapping a built-in under a
# name of your own is how docs/how-to/enable-built-in-checks.md says to
# configure one, so reading it as unfinished work made `hflow dataset create`
# select nothing at all on such a pipeline.
#
# SKIPPED is deliberately NOT here, and the distinction is load-bearing rather
# than pedantic. A step skipped because a critical check quarantined the
# episode is skipped CONDITIONALLY: retuning that check is the ordinary way to
# un-quarantine an episode, and the moment it passes, every step that stood
# aside has real work to do. Folding SKIPPED in here meant an un-quarantined
# episode never got its labels or its contact sheets on any later pass, while
# `hflow dataset create` -- whose quarantine rule now passed too -- shipped it
# as complete. That is why the engine records the two causes as two statuses
# (:class:`hflow.app.StepNotRun`) instead of one with a free-text reason.
#
# ERROR stays excluded for the same reason it is excluded above: a crash is
# infrastructure, so it is a retry, and a retry is work left to do.
SETTLED_STATUSES: tuple[CheckStatus, ...] = (*RAN_STATUSES, CheckStatus.SUPERSEDED)


@dataclass(frozen=True)
class Interval:
    """A labeled time span inside an episode, in nanoseconds of log time."""

    start_ns: int
    end_ns: int
    label: str = ""


@dataclass(frozen=True)
class Observation:
    """One structured observation aligned to an episode timestamp.

    ``observation_id`` is stable within the producing check (for example a
    frame index or annotation id). ``values`` carries scalar fields such as a
    reference, prediction, validity flag, latency, or raw response. The
    catalog stores one long-format row per value, keeping repeated evidence
    queryable without turning every sample into an episode-level column.
    Observations are for sparse check outputs; use a derived channel for dense
    numeric telemetry that belongs in the episode itself.
    """

    observation_id: str
    timestamp_ns: int
    values: dict[str, MeasurementValue]


@dataclass
class CheckResult:
    """What a check returns. Everything here lands in the episode's catalog row.

    :param measurements: Named numeric/string facts (catalog columns).
    :param observations: Repeated structured evidence aligned to episode
        timestamps (for example per-frame predictions and references).
    :param intervals: Labeled time spans (e.g. detected freezes).
    :param tags: Free-form labels routed to the catalog.
    :param verdict: Optional user-owned pass/fail. ``None`` means the check
        offers evidence only. A False verdict on a ``critical`` check
        quarantines the episode.
    """

    measurements: dict[str, MeasurementValue] = field(default_factory=dict)
    observations: list[Observation] = field(default_factory=list)
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
        if isinstance(self.value, bool):
            raise ValueError(f"threshold value for {self.key_pattern!r} must be a number, not bool")
        value_is_integer = isinstance(self.value, int)
        if not value_is_integer and math.isnan(self.value):
            raise ValueError(
                f"threshold value for {self.key_pattern!r} must be a number, not NaN: "
                "every comparison against NaN is False, so this gate would reject "
                "every episode it evaluated"
            )
        if not value_is_integer and math.isinf(self.value):
            accepts_every_finite_measurement = (
                self.comparison is Comparison.AT_MOST and self.value > 0
            ) or (self.comparison is Comparison.AT_LEAST and self.value < 0)
            outcome = (
                "accept every finite measurement"
                if accepts_every_finite_measurement
                else "reject every finite measurement"
            )
            raise ValueError(
                f"threshold value for {self.key_pattern!r} must be finite, not "
                f"{self.value}: this comparison would {outcome}"
            )

    def holds(self, measurement: float) -> bool:
        match self.comparison:
            case Comparison.AT_MOST:
                return measurement <= self.value
            case Comparison.AT_LEAST:
                return measurement >= self.value


@dataclass(frozen=True)
class Gate:
    """A verdict policy a pipeline attaches to one check: ``@app.check(version="1", gate=...)``.

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
    labels, video captions, and segmentations. They never gate anything -- gate semantics belong to critical
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
    on ``topic``. ``version`` is the pipeline author's compatibility promise,
    stamped into ``provenance/v1`` so an explicitly re-versioned derivation is
    a new pipeline and episode identity, never a silent rewrite.
    """

    topic: str
    function: DerivedFunction
    version: StepVersion


@dataclass(frozen=True)
class RegisteredEnrichment:
    """An enrichment function plus its registration configuration."""

    name: str
    function: EnrichmentFunction
    requires: frozenset[str]
    version: StepVersion


@dataclass(frozen=True)
class RegisteredCheck:
    """A check function plus its registration configuration.

    ``version`` is explicitly owned by the pipeline author. Bumping it makes a
    changed check append new-version measurement rows instead of mixing results
    that the author no longer considers comparable.
    """

    name: str
    function: CheckFunction
    critical: bool
    requires: frozenset[str]
    version: StepVersion
    gate: Gate | None = None
