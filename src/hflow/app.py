"""The App: check registration and the in-process dev loop.

``app.test(episode)`` is the dev loop: transform one input file into a
canonical episode, run every registered check and enrichment in-process --
no Docker, no scheduler -- and report measurements. ``app.run()`` provisions
the Docker Compose Airflow runtime (``hflow.runtime``) for durable,
observable batch ingestion of the same pipeline.

Ordering and gates follow docs/ARCHITECTURE.md: steps declaring resources
(``requires``/``uses``) run after plain ones (cheap-first), and a failed
verdict on a ``critical`` check quarantines the episode -- a tag, never a
deletion -- and skips its downstream steps. A check *crashing* is
infrastructure, not data: it is reported as an error, never recorded as a
quality outcome.
"""

import hashlib
import inspect
import json
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hflow.runtime import BundlePaths

from hflow.catalog import (
    AppendResult,
    Catalog,
    CheckRunRow,
    QuarantineHistory,
    content_episode_id,
)
from hflow.episode import Episode, _sanitize_topic
from hflow.ffmpeg import contact_sheet
from hflow.manifest import (
    DerivedChannelManifest,
    PipelineManifest,
    StepManifest,
)
from hflow.resample import DerivedSeries
from hflow.steps import (
    GATE_UNEVALUATED_TAG_PREFIX,
    RUN_PROFILES,
    CheckFunction,
    CheckResult,
    CheckStatus,
    DerivedChannel,
    DerivedFunction,
    EnrichmentFunction,
    EnrichmentResult,
    Gate,
    GateAbstained,
    GateDecided,
    RegisteredCheck,
    RegisteredEnrichment,
    Stage,
    compute_check_version,
    evaluate_gate,
    stages_for_profile,
)
from hflow.storage import (
    BucketStorageRoot,
    LocalStorageRoot,
    StorageRoot,
    fetch_uri,
    is_bucket_url,
    parse_storage_root,
)
from hflow.transform import (
    EpisodeStamps,
    TransformConfig,
    compute_pipeline_version,
    stamps_from_provenance,
    write_canonical_episode,
)
from hflow.workspace import RUNTIME_BUNDLE_DIRECTORY_NAME, Workspace

# The contract a @app.transform override implements: (source, output, config)
# -> the stamps it wrote. See :meth:`App.transform`.
TransformFunction = Callable[[Path, Path, TransformConfig], EpisodeStamps]

# The environment override for an App constructed without an explicit data
# root: the runtime (or a control plane provisioning a hosted workspace)
# exports it, and the same pipeline file runs unedited at every vantage --
# dev laptop, container mount, or per-workspace bucket prefix.
DATA_ROOT_ENVIRONMENT_VARIABLE = "HFLOW_DATA_ROOT"
DEFAULT_DATA_ROOT = "./data"

# Environment overrides for endpoint aliases (see App(endpoints=...)):
# HFLOW_ENDPOINT_<ALIAS>, the alias uppercased with every non-alphanumeric
# character replaced by "_". The deployment exports the variable; the
# pipeline file stays environment-portable.
ENDPOINT_ENVIRONMENT_VARIABLE_PREFIX = "HFLOW_ENDPOINT_"


def endpoint_environment_variable_name(alias: str) -> str:
    """The environment variable that overrides one endpoint alias."""
    sanitized_alias = re.sub(r"[^A-Za-z0-9]", "_", alias).upper()
    return f"{ENDPOINT_ENVIRONMENT_VARIABLE_PREFIX}{sanitized_alias}"


def _resolve_data_root(data_root: "Path | str | StorageRoot | None") -> "Path | str | StorageRoot":
    """Resolve the App's data root at the construction boundary.

    An explicit argument always wins; ``None`` means "let the environment
    decide": :data:`DATA_ROOT_ENVIRONMENT_VARIABLE` if set (how a runtime or
    control plane injects the workspace's root), else ``./data`` (the
    historical local default).
    """
    if data_root is not None:
        return data_root
    environment_data_root = os.environ.get(DATA_ROOT_ENVIRONMENT_VARIABLE)
    return environment_data_root if environment_data_root else DEFAULT_DATA_ROOT


# The ingest stage graph's "Media" sub-DAG collapsed to its v1 built-in: one
# contact sheet per camera topic, recorded exactly like an enrichment so its
# catalog rows flow through CheckRunRow like everything else.
MEDIA_CONTACT_SHEET_STEP_NAME = "media/contact_sheet"
# Published artifacts are recorded as measurements under this prefix, so a
# reader can tell "here is where the file went" from an ordinary label.
ARTIFACT_MEASUREMENT_KEY_PREFIX = "artifact/"
_MEDIA_CONTACT_SHEET_FPS = 0.5
_SYNC_COMPLETION_MARKER_NAME = ".sync-complete.json"


@dataclass(frozen=True)
class _SyncCompletion:
    """Proof that sync completed for one source path and canonical version."""

    source_path: str
    schema_version: str
    pipeline_version: str


def _source_identity(source_reference: Path | str, storage_root: StorageRoot) -> str:
    """Stable identity for one source episode reference.

    References under the data root identify by their ROOT-RELATIVE key, so
    the same episode named from different vantage points -- host path vs
    container mount of one directory, full ``gs://`` URL vs bucket key --
    always yields one identity (and therefore one run directory and one
    sync-completion lineage). Only sources outside any root fall back to the
    resolved absolute path or full URL, which are inherently
    vantage-specific.
    """
    if isinstance(source_reference, str) and is_bucket_url(source_reference):
        normalized_url = source_reference.rstrip("/")
        if isinstance(storage_root, BucketStorageRoot) and normalized_url.startswith(
            storage_root.url + "/"
        ):
            return normalized_url[len(storage_root.url) + 1 :]
        return normalized_url
    source_path = Path(source_reference)
    match storage_root:
        case BucketStorageRoot():
            if not source_path.is_absolute() and not source_path.is_file():
                return source_path.as_posix()  # a key under the bucket root
            return str(source_path.resolve())
        case LocalStorageRoot(path=root_path):
            try:
                return source_path.resolve().relative_to(root_path.resolve()).as_posix()
            except ValueError:
                return str(source_path.resolve())


def _source_artifact_directory_name(source_reference: Path | str, storage_root: StorageRoot) -> str:
    """A readable, collision-resistant directory name for one source.

    Source basenames are not identities: independent robots commonly produce
    files named ``run.mcap``. The source identity (root-relative key, or the
    absolute path/URL for sources outside the root) keeps their durable
    outputs disjoint without leaking the entire source hierarchy into the
    data root.
    """
    source_identifier = _source_identity(source_reference, storage_root)
    source_path_digest = hashlib.sha256(source_identifier.encode()).hexdigest()[:12]
    source_stem = Path(str(source_reference).rstrip("/")).stem
    return f"{source_stem}-{source_path_digest}"


def _write_sync_completion_marker(marker_path: Path, completion: _SyncCompletion) -> None:
    """Atomically publish proof that the canonical episode passed validation."""
    marker_payload = {
        "source_path": completion.source_path,
        "schema_version": completion.schema_version,
        "pipeline_version": completion.pipeline_version,
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=marker_path.parent,
        prefix=f".{marker_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_marker_stream:
        json.dump(marker_payload, temporary_marker_stream, sort_keys=True)
        temporary_marker_stream.write("\n")
        temporary_marker_path = Path(temporary_marker_stream.name)
    try:
        temporary_marker_path.replace(marker_path)
    finally:
        temporary_marker_path.unlink(missing_ok=True)


def _read_sync_completion_marker(marker_path: Path) -> _SyncCompletion:
    """Parse a sync completion marker into its refined internal shape."""
    if not marker_path.is_file():
        raise FileNotFoundError(
            f"canonical episode has no sync completion marker at {marker_path}; "
            "the previous sync may have failed -- run the sync or full profile again"
        )
    try:
        marker_payload = json.loads(marker_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid sync completion marker {marker_path}: {error}") from error
    if not isinstance(marker_payload, dict):
        raise ValueError(f"invalid sync completion marker {marker_path}: expected a JSON object")
    required_values: dict[str, str] = {}
    for field_name in ("source_path", "schema_version", "pipeline_version"):
        field_value = marker_payload.get(field_name)
        if not isinstance(field_value, str) or not field_value:
            raise ValueError(
                f"invalid sync completion marker {marker_path}: "
                f"{field_name!r} must be a non-empty string"
            )
        required_values[field_name] = field_value
    return _SyncCompletion(**required_values)


def _render_contact_sheets(canonical_episode: Episode, media_directory: Path) -> EnrichmentResult:
    """The built-in derived-media step: a contact sheet per camera topic.

    Each sheet lands at ``<media_directory>/<sanitized_topic>.jpg`` and is
    recorded as an ``artifact/<topic>`` measurement on the
    ``media/contact_sheet`` step.
    """
    sheet_artifacts: dict[str, Path] = {}
    for camera_topic in canonical_episode.cameras:
        frames = canonical_episode.frames(camera_topic, fps=_MEDIA_CONTACT_SHEET_FPS)
        sheet_path = media_directory / f"{_sanitize_topic(camera_topic)}.jpg"
        contact_sheet(frames, sheet_path)
        sheet_artifacts[camera_topic] = sheet_path
    return EnrichmentResult(artifacts=sheet_artifacts)


def _resolve_stages(stages: Iterable[Stage] | str | None) -> frozenset[Stage]:
    """Parse the ``stages=`` boundary: profile name, explicit set, or default."""
    if stages is None:
        return RUN_PROFILES["full"]
    if isinstance(stages, str):
        return stages_for_profile(stages)
    return frozenset(Stage(stage) for stage in stages)


@dataclass
class CheckRunReport:
    """Outcome of one check invocation inside a test run."""

    check: RegisteredCheck
    result: CheckResult | None = None
    error: str | None = None
    skipped_reason: str | None = None
    duration_s: float = 0.0

    @property
    def status(self) -> CheckStatus:
        if self.skipped_reason is not None:
            return CheckStatus.SKIPPED
        if self.error is not None:
            return CheckStatus.ERROR
        if self.result is not None and self.result.verdict is False:
            return CheckStatus.FAILED
        if self.result is not None and self.result.verdict is True:
            return CheckStatus.PASSED
        return CheckStatus.MEASURED


@dataclass
class EnrichmentRunReport:
    """Outcome of one enrichment invocation inside a test run."""

    enrichment: RegisteredEnrichment
    result: EnrichmentResult | None = None
    error: str | None = None
    skipped_reason: str | None = None
    duration_s: float = 0.0
    artifact_uris: dict[str, str] = field(default_factory=dict)

    @property
    def status(self) -> CheckStatus:
        # Enrichments have no verdicts, so only three statuses apply.
        if self.skipped_reason is not None:
            return CheckStatus.SKIPPED
        if self.error is not None:
            return CheckStatus.ERROR
        return CheckStatus.MEASURED


def _unsatisfiable_check_parameters(
    function: Callable[..., object],
) -> tuple[list[str], bool]:
    """Required parameters a registered check could never receive.

    At run time a check is called with exactly one argument: the canonical
    episode (``registered.function(canonical_episode)``). A check is only
    satisfiable if it can accept that episode positionally and every other
    parameter is optional (has a default) or is a ``*args``/``**kwargs``
    sink. Parameters the runtime call cannot supply (required
    positional-or-keyword beyond the episode, required keyword-only) would
    make every episode fail with a ``TypeError`` that the App records as an
    infrastructure error at ingest time.

    Returns ``(surplus, accepts_episode)``: the names of required parameters
    beyond the episode, and whether the signature has any slot that can
    receive the episode positionally (a plain parameter, or ``*args``, which
    absorbs it). ``**kwargs`` alone is not a slot: it cannot absorb a
    positional argument.

    A default on the first positional parameter does not stop it being the
    episode's slot. ``def check(episode=None)`` is called as
    ``check(canonical_episode)`` and the default is simply never used, so it
    is satisfiable and must keep registering.
    """
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return [], True
    seen_episode = False
    sees_varargs = False
    unsatisfiable: list[str] = []
    for parameter in parameters:
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            if not seen_episode:
                # The first positional slot takes the episode whether or not
                # it has a default, so claim it before testing for one.
                seen_episode = True
                continue
            if parameter.default is not inspect.Parameter.empty:
                continue
            unsatisfiable.append(parameter.name)
        elif parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            sees_varargs = True
        elif (
            parameter.kind is inspect.Parameter.KEYWORD_ONLY
            and parameter.default is inspect.Parameter.empty
        ):
            unsatisfiable.append(parameter.name)
    return unsatisfiable, seen_episode or sees_varargs


# Every step below is invoked with exactly one positional argument, an Episode:
# checks and enrichments with the canonical episode, derived-signal functions
# with the source episode. So all three share one satisfiability rule, and the
# only thing that differs is what the example in the message should say.
_STEP_EXAMPLE_BY_KIND = {
    # step kind -> (return annotation, wrapper-name suffix)
    "check": ("hflow.CheckResult", "check"),
    "enrichment": ("hflow.EnrichmentResult", "enrichment"),
    "derived channel": ("hflow.DerivedSeries", "series"),
}


def _raise_if_step_cannot_take_only_an_episode(
    function: Callable[..., object],
    *,
    step_kind: str,
    step_name: str,
    decorator: str,
) -> None:
    """Refuse at registration a step the runtime could never call.

    Registration is the only place this is knowable and the only place it is
    cheap: left alone, an unsatisfiable signature raises ``TypeError`` once
    per episode, which the App records as an infrastructure error while the
    measurement column goes quietly missing.
    """
    unsatisfiable, accepts_episode = _unsatisfiable_check_parameters(function)
    return_type, wrapper_suffix = _STEP_EXAMPLE_BY_KIND[step_kind]
    # A derived channel is named by its topic ("/joint_states"), which is not a
    # valid identifier, so the pasteable example needs a sanitized wrapper name.
    identifier_stem = re.sub(r"\W+", "_", step_name).strip("_")
    wrapper_name = f"{identifier_stem}_{wrapper_suffix}"
    if unsatisfiable:
        # Bind every unsatisfiable parameter, not just the first: the example is
        # meant to be pasted, and a snippet binding one of two still fails.
        sorted_names = sorted(unsatisfiable)
        bindings = ", ".join(f"{parameter_name}=..." for parameter_name in sorted_names)
        raise ValueError(
            f"{step_kind} {step_name!r} cannot be called with only an episode: "
            f"required parameter(s) without defaults: {', '.join(sorted_names)}. "
            "Wrap it in a function that binds them, e.g.\n\n"
            f"    {decorator}\n"
            f"    def {wrapper_name}(ep: hflow.Episode) -> {return_type}:\n"
            f"        return {getattr(function, '__name__', step_name)}(ep, {bindings})\n"
        )
    if not accepts_episode:
        raise ValueError(
            f"{step_kind} {step_name!r} cannot accept the episode: it must take "
            "the episode as its first positional parameter, e.g.\n\n"
            f"    {decorator}\n"
            f"    def {wrapper_name}(ep: hflow.Episode) -> {return_type}:\n"
            "        ...\n"
        )


def _execute_enrichment(
    registered_enrichment: RegisteredEnrichment,
    canonical_episode: Episode,
    skipped_reason: str | None,
) -> EnrichmentRunReport:
    """Run one enrichment-shaped step (user enrichment or the built-in media
    step) with the shared timing/boundary/error mechanics."""
    enrichment_run = EnrichmentRunReport(enrichment=registered_enrichment)
    if skipped_reason is not None:
        enrichment_run.skipped_reason = skipped_reason
        return enrichment_run
    started = time.perf_counter()
    try:
        returned_enrichment = registered_enrichment.function(canonical_episode)
        if isinstance(returned_enrichment, EnrichmentResult):
            enrichment_run.result = returned_enrichment
        else:
            enrichment_run.error = (
                f"enrichment returned {type(returned_enrichment).__name__}, "
                "expected hflow.EnrichmentResult -- wrap it: return "
                "hflow.EnrichmentResult(labels=...)"
            )
    except Exception:
        enrichment_run.error = traceback.format_exc(limit=8)
    finally:
        enrichment_run.duration_s = time.perf_counter() - started
    return enrichment_run


def _check_run_rows(report: "TestReport") -> list[CheckRunRow]:
    """The catalog rows one processed episode records: every check, then every
    enrichment's labels and published artifact keys.

    One owner, so the collision guard below and the catalog append can never
    disagree about what would be written. Artifact keys come from
    ``artifact_uris`` rather than the result's declared artifacts: a step whose
    artifact failed to publish contributes no key.
    """
    check_rows = [
        CheckRunRow.from_result(
            check_name=run.check.name,
            check_version=run.check.version,
            critical=run.check.critical,
            status=run.status,
            duration_s=run.duration_s,
            error=run.error,
            result=run.result,
        )
        for run in report.checks
    ]
    _raise_if_measurement_keys_claim_artifact_namespace(
        (row.check_name, key) for row in check_rows for key in row.measurements
    )
    for enrichment_run in report.enrichments:
        enrichment_result = enrichment_run.result
        labels: dict[str, float | int | str | bool] = (
            dict(enrichment_result.labels) if enrichment_result is not None else {}
        )
        if enrichment_result is not None:
            _raise_if_measurement_keys_claim_artifact_namespace(
                (enrichment_run.enrichment.name, key) for key in labels
            )
            labels.update(
                {
                    f"{ARTIFACT_MEASUREMENT_KEY_PREFIX}{artifact_name}": artifact_uri
                    for artifact_name, artifact_uri in enrichment_run.artifact_uris.items()
                }
            )
        check_rows.append(
            CheckRunRow(
                check_name=enrichment_run.enrichment.name,
                check_version=enrichment_run.enrichment.version,
                critical=False,
                status=enrichment_run.status,
                duration_s=enrichment_run.duration_s,
                error=enrichment_run.error,
                measurements=labels,
                tags=list(enrichment_result.tags) if enrichment_result is not None else [],
            )
        )
    return check_rows


def _raise_if_measurement_keys_claim_artifact_namespace(
    claimed: Iterable[tuple[str, str]],
) -> None:
    """Refuse a key that claims the framework's ``artifact/`` namespace.

    ``ARTIFACT_MEASUREMENT_KEY_PREFIX`` is reserved for URIs the framework
    itself publishes: this module packs them into the same measurements dict
    under that prefix, and snapshot.py exports every key under it as media.
    A user label reaching that dict under the prefix is indistinguishable
    from a real published artifact, and the ``labels.update`` merge below
    would let a real artifact of the same name silently overwrite the label.
    The enrichment call site runs before that merge, while the two are still
    told apart; the check call site runs on rows whose measurements never
    carry framework keys at all.
    """
    claimed_names = [
        f"{key!r} from {check_name!r}"
        for check_name, key in claimed
        if key.startswith(ARTIFACT_MEASUREMENT_KEY_PREFIX)
    ]
    if not claimed_names:
        return
    raise ValueError(
        f"measurement keys claim the framework's artifact/ namespace: "
        f"{'; '.join(claimed_names)}. Keys under artifact/ are reserved for the "
        "URIs the framework publishes (snapshot.py exports every such key as "
        "media), so a real artifact of the same name would silently overwrite "
        "the label. Rename the key."
    )


def _raise_if_measurement_keys_collide(check_rows: Sequence[CheckRunRow]) -> None:
    """Refuse a run in which two steps recorded the same measurement key.

    Curation ranks measurement rows per ``(episode_id, key)`` ordered by the
    owning episode's ``recorded_at`` then ``run_fingerprint``, and every step of
    ONE run shares both -- so two steps emitting one key on one episode is a
    total tie, and one row is dropped arbitrarily from ``measurements_latest``
    and therefore from the wide ``episodes`` view whose columns are enumerated
    from it. The surviving value is then attributed to whichever step the reader
    assumes.

    Keys are built from run-time topic names, so registration cannot know them;
    this is the one place a single episode's checks, enrichment labels, and
    published artifact keys are all visible together. Tags and intervals are
    deliberately unguarded: those tables carry ``check_name`` and have no
    per-key latest ranking, so two steps sharing one loses nothing.
    """
    steps_by_key: dict[str, list[str]] = {}
    for row in check_rows:
        for key in row.measurements:
            steps_by_key.setdefault(key, []).append(row.check_name)
    collisions = {key: names for key, names in steps_by_key.items() if len(names) > 1}
    if not collisions:
        return
    described = "; ".join(
        f"{key!r} from {', '.join(repr(name) for name in names)}"
        for key, names in sorted(collisions.items())
    )
    # Name a step the caller can actually edit: the built-in media step is not
    # theirs to rename, so prefer any other producer for the pasteable fix.
    producers = collisions[sorted(collisions)[0]]
    example_step = next(
        (name for name in reversed(producers) if name != MEDIA_CONTACT_SHEET_STEP_NAME),
        producers[-1],
    )
    example_function = re.sub(r"\W+", "_", example_step).strip("_") or "renamed_step"
    raise ValueError(
        f"steps recorded the same measurement key on one episode: {described}. "
        "Every step of one run shares its run_fingerprint and recorded_at, so the "
        "catalog ranks these rows as a tie and one of them silently disappears from "
        "measurements_latest and from the wide episodes view. If both steps are "
        "meant to run, give one its own key namespace; if one is a duplicate "
        "registration, drop it. Namespacing looks like:\n\n"
        f"@app.check(name={example_step!r})\n"
        f"def {example_function}(ep: hflow.Episode) -> hflow.CheckResult:\n"
        f"    result = ...  # what {example_step!r} computes today\n"
        "    return hflow.CheckResult(\n"
        f'        measurements={{f"{example_step}/{{key}}": value\n'
        "                      for key, value in result.measurements.items()},\n"
        "        intervals=result.intervals,\n"
        "        tags=result.tags,\n"
        "    )\n"
    )


def _apply_gate(registered: RegisteredCheck, run: "CheckRunReport") -> None:
    """Turn a registered gate into the verdict the meta loop acts on.

    Evaluated here rather than inside user code: a threshold applied inside a
    check raises on a key this episode never produced, and the runner records
    that as an infrastructure error -- discarding every measurement the check
    had already computed. Out here the evidence is recorded either way, and a
    gate that cannot be evaluated abstains instead of quietly passing.
    """
    result = run.result
    if registered.gate is None or result is None or run.error is not None:
        return
    match evaluate_gate(registered.gate, result.measurements):
        case GateAbstained(unevaluated_patterns=patterns):
            result.tags.extend(f"{GATE_UNEVALUATED_TAG_PREFIX}{pattern}" for pattern in patterns)
        case GateDecided(verdict=gate_verdict):
            # AND with the check's own verdict: a gate is an additional accept
            # condition the pipeline attached, so it can tighten policy and can
            # never resurrect an episode the check itself rejected.
            result.verdict = (
                gate_verdict if result.verdict is None else (result.verdict and gate_verdict)
            )


def _status_mark(status: CheckStatus) -> str:
    match status:
        case CheckStatus.PASSED:
            return "+"
        case CheckStatus.FAILED:
            return "x"
        case CheckStatus.MEASURED:
            return "*"
        case CheckStatus.SKIPPED:
            return "-"
        case CheckStatus.ERROR:
            return "!"


@dataclass
class TestReport:
    """Everything ``app.test()`` produced for one episode."""

    source_path: Path
    canonical_path: Path
    stamps: EpisodeStamps
    stages_run: frozenset[Stage] = frozenset(Stage)
    checks: list[CheckRunReport] = field(default_factory=list)
    enrichments: list[EnrichmentRunReport] = field(default_factory=list)
    quarantine_tags: list[str] = field(default_factory=list)
    catalog_entry: AppendResult | None = None

    @property
    def quarantined(self) -> bool:
        return bool(self.quarantine_tags)

    @property
    def has_errors(self) -> bool:
        """Whether any enabled check or enrichment failed to execute correctly."""
        return any(run.status is CheckStatus.ERROR for run in self.checks) or any(
            run.status is CheckStatus.ERROR for run in self.enrichments
        )

    def _stages_line(self) -> str:
        # Stage members in stage-graph order (declaration order).
        ordered_stage_names = [stage.value for stage in Stage if stage in self.stages_run]
        stage_list = ", ".join(ordered_stage_names) if ordered_stage_names else "(none)"
        matching_profiles = [
            profile_name
            for profile_name, profile_stages in RUN_PROFILES.items()
            if profile_stages == self.stages_run
        ]
        if matching_profiles:
            return f"stages: {matching_profiles[0]} ({stage_list})"
        return f"stages: {stage_list}"

    def summary(self) -> str:
        lines = [
            f"episode: {self.source_path.name} -> {self.canonical_path}",
            self._stages_line(),
            f"stamps: schema_version={self.stamps.schema_version} "
            f"pipeline_version={self.stamps.pipeline_version} "
            f"ffmpeg={self.stamps.ffmpeg_version}",
        ]
        if self.catalog_entry is not None:
            record_verb = "recorded" if self.catalog_entry.written else "already recorded"
            lines.append(
                f"catalog: episode {self.catalog_entry.episode_id} {record_verb} "
                f"(run {self.catalog_entry.run_fingerprint})"
            )
        if self.quarantine_tags:
            lines.append(f"QUARANTINED: {', '.join(self.quarantine_tags)}")
        lines.append("checks:")
        for run in self.checks:
            mark = _status_mark(run.status)
            headline = f"  {mark} {run.check.name} [{run.status}] ({run.duration_s * 1000:.0f}ms)"
            if run.skipped_reason is not None:
                lines.append(f"  {mark} {run.check.name} [skipped] {run.skipped_reason}")
                continue
            if run.error is not None:
                lines.append(f"{headline} {run.error}")
                continue
            lines.append(headline)
            if run.result is not None:
                for key, value in run.result.measurements.items():
                    formatted = f"{value:.6g}" if isinstance(value, float) else str(value)
                    lines.append(f"      {key} = {formatted}")
                for interval in run.result.intervals:
                    span_s = (interval.end_ns - interval.start_ns) / 1e9
                    lines.append(f"      interval {interval.label or '(unlabeled)'}: {span_s:.3f}s")
                if run.result.tags:
                    lines.append(f"      tags: {', '.join(run.result.tags)}")
        if self.enrichments:
            lines.append("enrichments:")
            for enrichment_run in self.enrichments:
                mark = _status_mark(enrichment_run.status)
                name = enrichment_run.enrichment.name
                if enrichment_run.skipped_reason is not None:
                    lines.append(f"  {mark} {name} [skipped] {enrichment_run.skipped_reason}")
                    continue
                headline = (
                    f"  {mark} {name} [{enrichment_run.status}] "
                    f"({enrichment_run.duration_s * 1000:.0f}ms)"
                )
                if enrichment_run.error is not None:
                    lines.append(f"{headline} {enrichment_run.error}")
                    continue
                lines.append(headline)
                if enrichment_run.result is not None:
                    for key, value in enrichment_run.result.labels.items():
                        formatted = f"{value:.6g}" if isinstance(value, float) else str(value)
                        lines.append(f"      {key} = {formatted}")
                    for artifact_name, artifact_path in enrichment_run.result.artifacts.items():
                        artifact_location = enrichment_run.artifact_uris.get(
                            artifact_name, str(artifact_path)
                        )
                        lines.append(
                            f"      {ARTIFACT_MEASUREMENT_KEY_PREFIX}{artifact_name} = "
                            f"{artifact_location}"
                        )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


class App:
    """A named pipeline: registered checks plus transform configuration.

    :param name: Pipeline name (display and DAG identity).
    :param data_root: Local directory or supported object-store prefix where
        runs write outputs (test runs land under ``<data_root>/test-runs/``).
        ``None`` (the default) resolves from the ``HFLOW_DATA_ROOT``
        environment variable when set, else ``./data`` -- so one pipeline
        file runs unedited on a laptop, inside the runtime containers, and
        in a hosted workspace whose root the deployment injects.
    :param transform: Canonical-transform configuration.
    :param endpoints: Named endpoint aliases (e.g. ``{"judge": "http://..."}``)
        that checks declare with ``uses="judge"`` and resolve via
        ``app.endpoints["judge"]`` in their own client code. At run start,
        an ``HFLOW_ENDPOINT_<ALIAS>`` environment variable overrides (or
        supplies) an alias's value, so deployments inject their own endpoints
        without editing pipeline code. The resolved ``app.endpoints`` mapping
        is read-only and rebuilt at every run start -- supply or override
        aliases through this parameter or the environment variable, never by
        mutating the mapping. (Named ``endpoints``, not "providers": in the
        Airflow ecosystem "provider" means a plugin package, and
        ``hflow.providers`` already means the video-protocol extension
        point.)
    """

    def __init__(
        self,
        name: str,
        data_root: Path | str | StorageRoot | None = None,
        *,
        transform: TransformConfig | None = None,
        endpoints: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.storage_root = parse_storage_root(_resolve_data_root(data_root))
        self.workspace = Workspace(self.storage_root)
        self.data_root: Path | str = (
            self.storage_root.path
            if isinstance(self.storage_root, LocalStorageRoot)
            else self.storage_root.url
        )
        self.transform_config = transform if transform is not None else TransformConfig()
        # The constructor literals stay pristine; ``endpoints`` is the
        # RESOLVED mapping, rebuilt (literals overlaid with the current
        # environment) at every run start so overrides set, changed, or
        # unset between runs all take effect. Read-only on purpose: a direct
        # mutation would be silently discarded by the next rebuild, so it
        # refuses loudly instead -- supply aliases via App(endpoints=...) or
        # HFLOW_ENDPOINT_<ALIAS>.
        self._endpoint_literals: dict[str, str] = dict(endpoints) if endpoints else {}
        self.endpoints: Mapping[str, str] = MappingProxyType(dict(self._endpoint_literals))
        self.checks: list[RegisteredCheck] = []
        self.enrichments: list[RegisteredEnrichment] = []
        self.derived: list[DerivedChannel] = []
        self.transform_override: TransformFunction | None = None

    def _registered_step_names(self) -> set[str]:
        # Checks, enrichments, and the built-in media step share the catalog's
        # check_name column, so names are unique across all three.
        return (
            {MEDIA_CONTACT_SHEET_STEP_NAME}
            | {registered.name for registered in self.checks}
            | {registered.name for registered in self.enrichments}
        )

    @property
    def pipeline_version(self) -> str:
        """The ``pipeline_version`` this App stamps onto episodes it processes.

        A content hash of the transform configuration plus the registered
        derived channels -- compare it against the catalog to find stale
        episodes (``hflow.stale_episodes`` / ``hflow stale``). A
        ``@app.transform`` override that computes its own derived channels
        owns its stamps; compare against a freshly stamped episode instead.
        """
        return compute_pipeline_version(
            self.transform_config,
            {channel.topic: channel.version for channel in self.derived},
        )

    def manifest(self) -> PipelineManifest:
        """This pipeline's JSON-able description: step names, content-hash
        versions, gate flags, endpoint aliases, and version stamps.

        The metadata a pipeline crosses a control boundary as (`hflow
        manifest` on the CLI): a service can display, diff, and validate
        pipelines from it without holding the code. Producing it requires
        importing the pipeline (versions hash the live functions), so
        generate it in the pipeline author's own environment and treat the
        result as the author's claims.
        """
        from hflow import __version__
        from hflow.format import EPISODE_FORMAT_VERSION

        return PipelineManifest(
            pipeline_name=self.name,
            hflow_version=__version__,
            schema_version=EPISODE_FORMAT_VERSION,
            pipeline_version=self.pipeline_version,
            checks=tuple(
                StepManifest.from_registered_check(registered) for registered in self.checks
            ),
            enrichments=tuple(
                StepManifest.from_registered_enrichment(registered)
                for registered in self.enrichments
            ),
            derived_channels=tuple(
                DerivedChannelManifest(topic=channel.topic, version=channel.version)
                for channel in self.derived
            ),
            endpoint_aliases=tuple(
                sorted(set(self._endpoint_literals) | self._used_endpoint_aliases())
            ),
            has_transform_override=self.transform_override is not None,
        )

    def check(
        self,
        *,
        name: str | None = None,
        critical: bool = False,
        requires: Iterable[str] | None = None,
        uses: str | None = None,
        gate: Gate | None = None,
        version: str | None = None,
    ) -> Callable[[CheckFunction], CheckFunction]:
        """Register a check function. See ``hflow.steps.CheckResult``.

        ``gate`` attaches a pass/fail policy the runner evaluates over the
        measurements this check returns, so a built-in that records evidence
        only can gate without being rewritten. HFlow ships recommended gates
        as values (``hflow.checks.RECOMMENDED_CAMERA_INTEGRITY``); pass one, or
        a copy with your own numbers. Combined with ``critical=True`` a failing
        gate quarantines; without it, the run proceeds with a ``failed:`` tag.

        ``version`` explicitly identifies opaque configuration that cannot be
        derived from function source, defaults, or captured stable values.
        """

        def register(function: CheckFunction) -> CheckFunction:
            check_name = name if name is not None else getattr(function, "__name__", "")
            if not check_name:
                raise ValueError("pass name=... when registering a callable without __name__")
            if check_name in self._registered_step_names():
                raise ValueError(f"a step named {check_name!r} is already registered")
            _raise_if_step_cannot_take_only_an_episode(
                function, step_kind="check", step_name=check_name, decorator="@app.check()"
            )
            if gate is not None and not isinstance(gate, Gate):
                raise ValueError(
                    f"check {check_name!r} was registered with gate="
                    f"{type(gate).__name__}, expected an hflow.Gate -- a gate is a value "
                    "you build once and pass in, e.g.\n\n"
                    "    @app.check(critical=True, "
                    "gate=hflow.checks.RECOMMENDED_CAMERA_INTEGRITY)\n"
                    f"    def {check_name}(ep: hflow.Episode) -> hflow.CheckResult:\n"
                    "        return hflow.checks.camera_frame_stats(ep)\n"
                )
            requires_set = frozenset(requires) if requires is not None else frozenset()
            self.checks.append(
                RegisteredCheck(
                    name=check_name,
                    function=function,
                    critical=critical,
                    requires=requires_set,
                    uses=uses,
                    version=compute_check_version(
                        check_name,
                        function,
                        critical,
                        requires_set,
                        uses,
                        version,
                        gate=gate,
                    ),
                    gate=gate,
                )
            )
            return function

        return register

    def enrich(
        self,
        *,
        name: str | None = None,
        requires: Iterable[str] | None = None,
        uses: str | None = None,
        version: str | None = None,
    ) -> Callable[[EnrichmentFunction], EnrichmentFunction]:
        """Register an enrichment. See ``hflow.steps.EnrichmentResult``.

        Enrichments run after every check and never on a quarantined episode
        (the gate semantics: no enrichment spend on bad data).
        """

        def register(function: EnrichmentFunction) -> EnrichmentFunction:
            enrichment_name = name if name is not None else getattr(function, "__name__", "")
            if not enrichment_name:
                raise ValueError("pass name=... when registering a callable without __name__")
            if enrichment_name in self._registered_step_names():
                raise ValueError(f"a step named {enrichment_name!r} is already registered")
            _raise_if_step_cannot_take_only_an_episode(
                function,
                step_kind="enrichment",
                step_name=enrichment_name,
                decorator="@app.enrich()",
            )
            requires_set = frozenset(requires) if requires is not None else frozenset()
            self.enrichments.append(
                RegisteredEnrichment(
                    name=enrichment_name,
                    function=function,
                    requires=requires_set,
                    uses=uses,
                    version=compute_check_version(
                        enrichment_name,
                        function,
                        False,
                        requires_set,
                        uses,
                        version,
                    ),
                )
            )
            return function

        return register

    def derive(
        self, topic: str, *, version: str | None = None
    ) -> Callable[[DerivedFunction], DerivedFunction]:
        """Register a derived-signal function: ``(Episode) -> DerivedSeries``.

        The function is computed over the SOURCE episode during
        :meth:`process` (before the canonical transform; ``Episode`` works on
        pre-canonical files for state channels) and its series is written as a
        new JSON channel on ``topic`` in the canonical file. Build series with
        ``hflow.to_grid`` or construct a ``DerivedSeries`` directly.
        """
        if not topic:
            raise ValueError(f"derived channel topic must not be empty, got {topic!r}")

        def register(function: DerivedFunction) -> DerivedFunction:
            if any(registered.topic == topic for registered in self.derived):
                raise ValueError(f"a derived channel for topic {topic!r} is already registered")
            _raise_if_step_cannot_take_only_an_episode(
                function,
                step_kind="derived channel",
                step_name=topic,
                decorator=f'@app.derive("{topic}")',
            )
            self.derived.append(
                DerivedChannel(
                    topic=topic,
                    function=function,
                    version=compute_check_version(
                        topic,
                        function,
                        False,
                        frozenset(),
                        None,
                        version,
                    ),
                )
            )
            return function

        return register

    def transform(self, function: TransformFunction) -> TransformFunction:
        """Replace the default canonical transform (at most one override).

        The override is called as ``function(source, output, config)`` and
        must return the :class:`EpisodeStamps` it wrote. Contract: it must
        still end by calling :func:`write_canonical_episode` so every
        downstream contract (canonical video, chunk groups, ``provenance/v1``)
        holds. The override receives no derived plumbing -- :meth:`process`
        auto-computes and passes ``derived=`` only for the DEFAULT transform;
        an override that wants derived channels calls
        ``write_canonical_episode(derived=...)`` itself.
        """
        if self.transform_override is not None:
            raise ValueError("a transform override is already registered (at most one)")
        self.transform_override = function
        return function

    def _fetch_source(self, episode: Path | str) -> Path:
        """Resolve a source reference to a readable local file.

        Accepted forms, in resolution order: a full bucket URL (fetched
        through the data root's mirror when it lives under the root), an
        existing local path (the historical library behavior -- cwd-relative
        or absolute), and a key relative to the data root (the form ingest
        conf URIs arrive in, for local and bucket roots alike).
        """
        if isinstance(episode, str) and is_bucket_url(episode):
            normalized_url = episode.rstrip("/")
            if isinstance(self.storage_root, BucketStorageRoot) and normalized_url.startswith(
                self.storage_root.url + "/"
            ):
                return self.storage_root.fetch(normalized_url[len(self.storage_root.url) + 1 :])
            return fetch_uri(normalized_url)
        episode_path = Path(episode)
        if episode_path.is_file():
            return episode_path
        if not episode_path.is_absolute():
            match self.storage_root:
                case BucketStorageRoot() as bucket_root:
                    # Raises FileNotFoundError naming the full remote location.
                    return bucket_root.fetch(episode_path.as_posix())
                case LocalStorageRoot(path=root_path):
                    candidate = root_path / episode_path
                    if candidate.is_file():
                        return candidate
        raise FileNotFoundError(
            f"episode {str(episode)!r} not found: not an existing local file, and not "
            f"a key under the data root {self.storage_root}"
        )

    def _used_endpoint_aliases(self) -> set[str]:
        return {
            registered.uses
            for registered in [*self.checks, *self.enrichments]
            if registered.uses is not None
        }

    def _resolve_endpoint_overrides(self) -> None:
        """Rebuild ``endpoints``: literals overlaid with the environment.

        The environment wins over a literal in the pipeline file, and an
        alias supplied ONLY by the environment satisfies steps' ``uses=``
        preflight -- this is how a deployment (or a control plane) injects
        per-workspace endpoints without editing customer code. Runs at
        preflight, after every registration, so ``app.endpoints[alias]``
        inside a running step always sees the resolved value; rebuilding
        from the pristine literals each time means an override set, changed,
        or UNSET between runs in one process always takes effect.

        The environment naming is lossy (non-alphanumerics collapse to
        ``_``), so two aliases that map to one variable would be silently
        co-overridden -- refused loudly here instead.
        """
        aliases = sorted(set(self._endpoint_literals) | self._used_endpoint_aliases())
        aliases_by_variable: dict[str, list[str]] = {}
        for alias in aliases:
            aliases_by_variable.setdefault(endpoint_environment_variable_name(alias), []).append(
                alias
            )
        colliding = {
            variable: names for variable, names in aliases_by_variable.items() if len(names) > 1
        }
        if colliding:
            raise ValueError(
                "endpoint aliases are indistinguishable under HFLOW_ENDPOINT_* naming: "
                + "; ".join(
                    f"{variable} would override all of {names}"
                    for variable, names in sorted(colliding.items())
                )
                + " -- rename an alias so each maps to a distinct environment variable"
            )
        resolved = dict(self._endpoint_literals)
        for alias in aliases:
            environment_override = os.environ.get(endpoint_environment_variable_name(alias))
            if environment_override:
                resolved[alias] = environment_override
        self.endpoints = MappingProxyType(resolved)

    def _preflight(self) -> None:
        self._resolve_endpoint_overrides()
        missing = sorted(
            {alias for alias in self._used_endpoint_aliases() if alias not in self.endpoints}
        )
        if missing:
            raise ValueError(
                f"steps declare endpoint aliases {missing} but App(endpoints=...) "
                f"defines only {sorted(self.endpoints)} -- pass the alias there, or export "
                + ", ".join(endpoint_environment_variable_name(alias) for alias in missing)
            )

    def _ordered_checks(self) -> list[RegisteredCheck]:
        # Cheap-first: steps that need no special resources run before steps
        # declaring requires/uses. Stable within each class.
        return sorted(
            self.checks,
            key=lambda registered: bool(registered.requires) or registered.uses is not None,
        )

    def _ordered_enrichments(self) -> list[RegisteredEnrichment]:
        return sorted(
            self.enrichments,
            key=lambda registered: bool(registered.requires) or registered.uses is not None,
        )

    def test(
        self,
        episode: Path | str,
        *,
        output_dir: Path | str | StorageRoot | None = None,
        verbose: bool = True,
        record: bool = False,
        stages: Iterable[Stage] | str | None = None,
    ) -> TestReport:
        """The dev loop: run the whole pipeline on one episode, in-process.

        A thin wrapper over :meth:`process` that defaults to
        ``<data_root>/test-runs/<stem>-<source-identity-hash>/``, prints the
        summary, and does NOT
        record to the catalog -- iterating on a check should not pollute it.
        Pass ``record=True`` to append the run (idempotent per episode
        content and step versions). ``stages`` selects a run profile or an
        explicit stage set, exactly as in :meth:`process`.
        """
        return self.process(
            episode,
            output_dir=(
                output_dir
                if output_dir is not None
                else self.workspace.test_runs_root.child(
                    _source_artifact_directory_name(episode, self.storage_root)
                )
            ),
            verbose=verbose,
            record=record,
            stages=stages,
        )

    def run(
        self,
        pipeline_file: Path | str | None = None,
        *,
        bundle_dir: Path | str | None = None,
        hflow_source: Path | str | None = None,
        requirements_file: Path | str | None = None,
        wait_timeout_s: float = 300.0,
    ) -> "BundlePaths":
        """Provision the Compose runtime for this pipeline (Docker mode).

        Renders the runtime bundle into ``<data_root>/runtime`` (or
        ``bundle_dir``), starts it with ``docker compose up -d``, waits until
        Airflow reports healthy, and prints the UI URL, credentials, and
        ingest DAG id. The first start pulls images and builds the user venv
        (minutes); later starts reuse both.

        ``pipeline_file`` defaults to the executing script (the file that
        defines this App): call ``app.run()`` under an ``if __name__ ==
        "__main__":`` guard, because the runtime re-imports the same file to
        load the App -- an unguarded ``run()`` would start the runtime from
        inside its own worker. In notebooks/REPLs there is no script file, so
        pass ``pipeline_file=`` explicitly.
        """
        from hflow.runtime import RuntimeConfig, infer_hflow_source
        from hflow.runtime._lifecycle import start_runtime, started_summary

        app_variable = "app"
        if pipeline_file is None:
            main_module = sys.modules.get("__main__")
            main_file = getattr(main_module, "__file__", None)
            if main_file is None:
                raise RuntimeError(
                    "app.run() cannot infer the pipeline file: __main__ has no __file__ "
                    "(interactive or notebook session). Save the pipeline as a .py file "
                    "and pass app.run(pipeline_file='path/to/pipeline.py')."
                )
            pipeline_file = Path(main_file)
            # The runtime resolves the App by variable name inside the file;
            # find what this instance is actually called in the script.
            for module_variable, value in vars(main_module).items():
                if value is self:
                    app_variable = module_variable
                    break

        resolved_source = Path(hflow_source) if hflow_source is not None else infer_hflow_source()
        config = RuntimeConfig(
            pipeline_file=Path(pipeline_file),
            data_root=(
                self.storage_root.path
                if isinstance(self.storage_root, LocalStorageRoot)
                else self.storage_root.url
            ),
            app_variable=app_variable,
            requirements_file=Path(requirements_file) if requirements_file is not None else None,
            hflow_source=resolved_source,
        )
        # A bucket data root has no local directory to host the bundle, so it
        # defaults to ./runtime in the working directory instead.
        default_bundle_dir = (
            self.storage_root.path / RUNTIME_BUNDLE_DIRECTORY_NAME
            if isinstance(self.storage_root, LocalStorageRoot)
            else Path(RUNTIME_BUNDLE_DIRECTORY_NAME)
        )
        resolved_bundle_dir = Path(bundle_dir) if bundle_dir is not None else default_bundle_dir
        # Narrate the long provisioning stages to stderr (stdout stays the
        # final summary), matching the CLI's `hflow up` behavior.
        paths, _ = start_runtime(
            config,
            resolved_bundle_dir,
            wait_timeout_s=wait_timeout_s,
            on_progress=lambda event: print(event, file=sys.stderr),
        )
        print(started_summary(paths))
        return paths

    def process(
        self,
        episode: Path | str,
        *,
        output_dir: Path | str | StorageRoot | None = None,
        verbose: bool = False,
        record: bool = True,
        stages: Iterable[Stage] | str | None = None,
        quarantine_history: QuarantineHistory | None = None,
        orchestrator_run_id: str | None = None,
    ) -> TestReport:
        """Process one episode through the enabled stages of the stage
        graph: transform to canonical (``sync``), run checks with gate
        semantics (``meta``), run enrichments (``labels``), render derived
        media (``media``), and record whatever the enabled stages produced
        to the catalog.

        This is the operation the ingest DAG maps over episodes;
        :meth:`test` wraps it for the dev loop. Outputs land under
        ``<data_root>/episodes/<stem>-<source-identity-hash>/`` unless
        ``output_dir`` is given.

        ``stages`` is a run-profile name (see ``hflow.RUN_PROFILES``), an
        explicit stage set, or ``None`` for the full profile. Without
        ``sync``, the canonical file must already exist in the run dir and
        stamps are reconstructed from its own provenance record. Without
        ``meta``, the quarantine gate for ``labels``/``media`` comes from the
        episode's latest cataloged state (no catalog = no known quarantine).

        ``quarantine_history`` is that gate's catalog reader, open across a
        whole batch so a stage does not re-sync and re-open the catalog once
        per episode; omit it and this call opens one for itself.

        ``orchestrator_run_id`` records which orchestrated run produced the
        row, so "which run wrote this" is answerable from the catalog alone.
        Named for the role rather than for a scheduler: the generated Airflow
        DAGs pass their stage sub-DAG's own run id, which is the id the runs
        API reports for that stage, and another backend would pass its own
        equivalent. The dev loop passes nothing and records NULL. Provenance
        only, never part of any identity hash (see
        :meth:`hflow.catalog.Catalog.append_episode`).
        """
        enabled_stages = _resolve_stages(stages)
        source_identifier = _source_identity(episode, self.storage_root)
        self._preflight()
        if Stage.SYNC in enabled_stages:
            source_path = self._fetch_source(episode)  # the transform reads it
        else:
            # Non-sync stages read only the canonical episode: the raw source
            # is identity, not input. Never fetch it -- a relabel on a fresh
            # worker would otherwise download the full raw file just to
            # compute a name (the sync-completion marker still proves the
            # canonical belongs to this source).
            source_path = Path(str(episode).rstrip("/"))

        run_storage_root = (
            parse_storage_root(output_dir)
            if output_dir is not None
            else self.workspace.episodes_root.child(
                _source_artifact_directory_name(episode, self.storage_root)
            )
        )
        run_dir = run_storage_root.workspace
        run_dir.mkdir(parents=True, exist_ok=True)
        canonical_file_name = f"{source_path.stem}.canonical.mcap"
        canonical_path = run_dir / canonical_file_name
        scratch_dir = run_dir / "scratch"
        sync_completion_marker_path = run_dir / _SYNC_COMPLETION_MARKER_NAME

        stamps: EpisodeStamps | None = None
        sync_completion: _SyncCompletion | None = None
        if Stage.SYNC in enabled_stages:
            # File existence is not proof of a successful rewrite. Clear the
            # durable proof before starting so a tolerated sync failure cannot
            # let a later sub-DAG consume a previous canonical episode.
            run_storage_root.delete(_SYNC_COMPLETION_MARKER_NAME)
            if self.transform_override is not None:
                # Derived signals are the override's own responsibility (see
                # :meth:`transform`): no auto-computation, no derived plumbing.
                stamps = self.transform_override(source_path, canonical_path, self.transform_config)
                if not isinstance(stamps, EpisodeStamps):
                    raise TypeError(
                        f"transform override returned {type(stamps).__name__}, expected "
                        "hflow.EpisodeStamps -- return the stamps from write_canonical_episode"
                    )
            else:
                # Derived signals are computed over the SOURCE episode: they
                # resample raw state channels, which Episode reads pre-canonical.
                derived_series: list[tuple[str, DerivedSeries, str]] = []
                if self.derived:
                    with Episode(source_path) as source_episode:
                        for derived_channel in self.derived:
                            returned_series = derived_channel.function(source_episode)
                            if not isinstance(returned_series, DerivedSeries):
                                raise TypeError(
                                    f"derived function for topic {derived_channel.topic!r} "
                                    f"returned {type(returned_series).__name__}, expected "
                                    "hflow.DerivedSeries -- build one with hflow.to_grid(...)"
                                )
                            derived_series.append(
                                (derived_channel.topic, returned_series, derived_channel.version)
                            )
                stamps = write_canonical_episode(
                    source_path,
                    canonical_path,
                    self.transform_config,
                    source_uri=source_identifier,
                    derived=derived_series or None,
                )
            # The canonical file was just rewritten, so any mp4/frame artifacts
            # a previous run cached in the scratch dir are stale -- including
            # ones from a different source episode sharing this run dir's stem.
            if scratch_dir.exists():
                shutil.rmtree(scratch_dir)
        else:
            try:
                canonical_path = run_storage_root.fetch(canonical_file_name)
            except FileNotFoundError as error:
                raise FileNotFoundError(
                    f"stage set {sorted(stage.value for stage in enabled_stages)} omits "
                    f"'{Stage.SYNC}' but no canonical episode exists at {canonical_path} -- "
                    "run the full or sync profile first"
                ) from error
            try:
                sync_completion_marker_path = run_storage_root.fetch(_SYNC_COMPLETION_MARKER_NAME)
            except FileNotFoundError as error:
                raise FileNotFoundError(
                    "canonical episode has no sync completion marker at "
                    f"{run_storage_root.uri_for(_SYNC_COMPLETION_MARKER_NAME)}; "
                    "the previous sync may have failed -- run the sync or full profile again"
                ) from error
            sync_completion = _read_sync_completion_marker(sync_completion_marker_path)
            if sync_completion.source_path != source_identifier:
                raise ValueError(
                    f"sync completion marker {sync_completion_marker_path} belongs to "
                    f"{sync_completion.source_path!r}, not {source_identifier!r}; "
                    "run the sync or full profile again"
                )

        with Episode(canonical_path, workdir=scratch_dir) as canonical_episode:
            canonical_stamps = stamps_from_provenance(canonical_episode.metadata)
            if stamps is None:
                # sync did not run this invocation: the canonical file's own
                # provenance record is the one owner of its stamps.
                stamps = canonical_stamps
                assert sync_completion is not None
                if (
                    sync_completion.schema_version != stamps.schema_version
                    or sync_completion.pipeline_version != stamps.pipeline_version
                ):
                    raise ValueError(
                        f"sync completion marker {sync_completion_marker_path} does not "
                        "match the canonical episode's provenance; run the sync or full "
                        "profile again"
                    )
            elif stamps != canonical_stamps:
                raise ValueError(
                    "transform returned stamps that do not match the canonical episode's "
                    "provenance record"
                )

            if Stage.SYNC in enabled_stages:
                canonical_uri = run_storage_root.publish(canonical_path, canonical_file_name)
                _write_sync_completion_marker(
                    sync_completion_marker_path,
                    _SyncCompletion(
                        source_path=source_identifier,
                        schema_version=stamps.schema_version,
                        pipeline_version=stamps.pipeline_version,
                    ),
                )
                run_storage_root.publish(sync_completion_marker_path, _SYNC_COMPLETION_MARKER_NAME)
            else:
                canonical_uri = run_storage_root.uri_for(canonical_file_name)

            report = TestReport(
                source_path=source_path,
                canonical_path=canonical_path,
                stamps=stamps,
                stages_run=enabled_stages,
            )

            checks_to_run = self._ordered_checks() if Stage.META in enabled_stages else []
            for registered in checks_to_run:
                run = CheckRunReport(check=registered)
                report.checks.append(run)
                if report.quarantined:
                    run.skipped_reason = (
                        f"episode quarantined ({', '.join(report.quarantine_tags)})"
                    )
                    continue
                started = time.perf_counter()
                try:
                    returned = registered.function(canonical_episode)
                    # Parse the boundary: user code may return anything.
                    if isinstance(returned, CheckResult):
                        run.result = returned
                    else:
                        run.error = (
                            f"check returned {type(returned).__name__}, expected "
                            "hflow.CheckResult -- wrap it: return hflow.CheckResult("
                            "measurements=...)"
                        )
                except Exception:
                    # Infrastructure, not data: never recorded as a quality outcome.
                    run.error = traceback.format_exc(limit=8)
                finally:
                    run.duration_s = time.perf_counter() - started
                _apply_gate(registered, run)
                if run.result is not None and run.result.verdict is False:
                    if registered.critical:
                        report.quarantine_tags.append(f"quarantined:{registered.name}")
                    else:
                        run.result.tags.append(f"failed:{registered.name}")

            # Quarantine gate for labels/media: meta's in-memory result when
            # it ran in this same invocation; otherwise the episode's latest
            # cataloged state. No catalog row = no known quarantine: proceed.
            # A cataloged quarantine is carried into this run's tags so a
            # recorded run without meta never masks the state.
            if Stage.META not in enabled_stages:
                episode_id = content_episode_id(canonical_path)
                if quarantine_history is not None:
                    carried_tags = quarantine_history.quarantine_tags(episode_id)
                else:
                    with QuarantineHistory(self.workspace.catalog_root) as history:
                        carried_tags = history.quarantine_tags(episode_id)
                if carried_tags is not None:
                    report.quarantine_tags.extend(carried_tags)
            quarantine_skip_reason = (
                f"episode quarantined ({', '.join(report.quarantine_tags)})"
                if report.quarantined
                else None
            )

            if Stage.LABELS in enabled_stages:
                for registered_enrichment in self._ordered_enrichments():
                    report.enrichments.append(
                        _execute_enrichment(
                            registered_enrichment, canonical_episode, quarantine_skip_reason
                        )
                    )

            # The media stage is silently absent on a camera-less episode:
            # there is nothing to render, so no row claims otherwise.
            if Stage.MEDIA in enabled_stages and canonical_episode.cameras:
                media_directory = run_dir / "media"

                def render_contact_sheets(media_episode: Episode) -> EnrichmentResult:
                    return _render_contact_sheets(media_episode, media_directory)

                media_step = RegisteredEnrichment(
                    name=MEDIA_CONTACT_SHEET_STEP_NAME,
                    function=render_contact_sheets,
                    requires=frozenset(),
                    uses=None,
                    # Versioned by the implementation function, so a changed
                    # renderer is a new measurement identity.
                    version=compute_check_version(
                        MEDIA_CONTACT_SHEET_STEP_NAME,
                        _render_contact_sheets,
                        False,
                        frozenset(),
                        None,
                    ),
                )
                report.enrichments.append(
                    _execute_enrichment(media_step, canonical_episode, quarantine_skip_reason)
                )

            episode_metadata = dict(canonical_episode.metadata)

        for enrichment_run in report.enrichments:
            enrichment_result = enrichment_run.result
            if enrichment_result is None:
                continue
            for artifact_name, artifact_path in enrichment_result.artifacts.items():
                resolved_artifact_path = artifact_path.resolve()
                try:
                    artifact_relative_path = resolved_artifact_path.relative_to(run_dir.resolve())
                    artifact_key = artifact_relative_path.as_posix()
                except ValueError:
                    step_directory = (
                        f"{_sanitize_topic(enrichment_run.enrichment.name)}-"
                        f"{enrichment_run.enrichment.version}"
                    )
                    artifact_name_digest = hashlib.sha256(artifact_name.encode()).hexdigest()[:8]
                    artifact_key = (
                        f"artifacts/{step_directory}/{_sanitize_topic(artifact_name)}-"
                        f"{artifact_name_digest}/{artifact_path.name}"
                    )
                try:
                    enrichment_run.artifact_uris[artifact_name] = run_storage_root.publish(
                        artifact_path, artifact_key
                    )
                except Exception:
                    # A missing or unreadable artifact file is the STEP's
                    # failure (user code declared a path it never wrote), not
                    # the run's: record it like any other step error and keep
                    # every other step's completed results.
                    publish_error = (
                        f"artifact {artifact_name!r} at {artifact_path} could not be "
                        f"published:\n{traceback.format_exc(limit=4)}"
                    )
                    enrichment_run.error = (
                        f"{enrichment_run.error}\n{publish_error}"
                        if enrichment_run.error
                        else publish_error
                    )

        # Assembled even when not recording, so the dev loop refuses a key
        # collision on episode one instead of at the first curation query.
        check_rows = _check_run_rows(report)
        _raise_if_measurement_keys_collide(check_rows)
        if record:
            report.catalog_entry = Catalog(self.workspace.catalog_root).append_episode(
                canonical_path=canonical_path,
                uri=canonical_uri,
                stamps=stamps,
                episode_metadata=episode_metadata,
                check_rows=check_rows,
                quarantine_tags=report.quarantine_tags,
                source_uri=source_identifier,
                orchestrator_run_id=orchestrator_run_id,
            )

        if verbose:
            print(report.summary())
        return report


def parse_pipeline_spec(pipeline_spec: str) -> tuple[Path, str]:
    """Split ``path/to/pipeline.py[:app_variable]`` (default variable: ``app``)."""
    path_part, separator, variable_part = pipeline_spec.rpartition(":")
    if separator and path_part and variable_part.isidentifier():
        return Path(path_part), variable_part
    return Path(pipeline_spec), "app"


def import_pipeline_application(pipeline_spec: str) -> "App":
    """Import ``path/to/pipeline.py[:app]`` and return its :class:`App`, loudly.

    One owner for the "address a pipeline by file" contract every vantage
    needs -- the CLI's ``manifest``/``up``/``deploy``/``stale``, and any other
    caller that must hold a user's pipeline (the workspace UI's pipeline
    page). The pipeline file is arbitrary user code, so importing EXECUTES
    it: any exception it raises is a boundary failure reported as a
    ``ValueError`` naming the file, never a crash of the calling program.
    """
    import importlib.util

    pipeline_file, app_variable = parse_pipeline_spec(pipeline_spec)
    spec = importlib.util.spec_from_file_location("hflow_user_pipeline", pipeline_file)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import pipeline file {pipeline_file}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except (Exception, SystemExit) as error:
        # SystemExit is a BaseException: a pipeline that guards its config at
        # import time (sys.exit("set ROBOT_FLEET"), or a module-scope argparse)
        # would otherwise walk past `except Exception` and take the calling
        # program's exit status with it -- killing a long-lived UI server at
        # startup. KeyboardInterrupt stays uncaught on purpose: that one
        # belongs to whoever pressed it, not to the pipeline file.
        raise ValueError(f"importing {pipeline_file} failed: {error}") from error
    application = getattr(module, app_variable, None)
    if not isinstance(application, App):
        raise ValueError(f"{pipeline_file} has no hflow.App named {app_variable!r}")
    return application
