"""Which stages a recording still needs, asked of the catalog rather than redone.

Re-ingesting a corpus used to redo every stage on every episode: the
transcode, the ffmpeg decode the camera checks share, the contact sheets.
``sync`` already stopped redoing the transcode when the source bytes, the
pipeline version and the encoder all match what it last recorded (see
:meth:`hflow.App.process`). This module does the same for the rest of the
graph.

It is one question, asked per episode and per step: **has this step already
recorded an outcome against this exact episode content, at the version the
step has now?** Neither identity is ordered. ``episode_id`` hashes the
canonical bytes, so an episode the transform would now produce differently is
a different id and nothing filed under the old one counts. ``check_version``
is the pipeline author's explicit compatibility promise, so bumping it after a
retuned threshold makes the old rows stop counting. A step is current or it is
not.

Two design decisions carry the correctness of the whole module.

**The plan is computed after ``sync`` has run, never before.** Only then does
``episodes_latest`` name the episode this run actually produced, so the
outstanding-steps question is asked against the right content id. Planning
first would read the PREVIOUS generation's id, and would skip ``meta`` for a
recording whose source bytes had changed underneath it -- the one thing the
catalog genuinely cannot see, since nothing in it hashes the source. That also
removes any need to compare ``pipeline_version`` here: after sync, the episode
the catalog names for a source IS this run's, by construction.

**``sync`` itself is never planned away.** It is both the producer of the
canonical file every later stage reads and its own cache, so leaving it in
means the planner reasons only about whether WORK is current, never about
whether a FILE still exists -- which the catalog cannot answer either. A run
directory someone cleaned out costs one transcode, not a ``FileNotFoundError``
in the middle of the meta stage.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from hflow.steps import SETTLED_STATUSES, Stage

if TYPE_CHECKING:
    import duckdb

    from hflow.app import App


class StageSelection(StrEnum):
    """Whether a direct run re-does work the catalog already records as current.

    A run's own vocabulary rather than a flag, because the two modes are
    genuinely different requests and a caller should have to name which one it
    means. It stays a direct-run type even though the scheduled lane now plans
    too: what crosses the conf boundary there is one boolean per sub-DAG run
    (``all_stages``), not a selection between named modes, and widening this
    enum into conf vocabulary would oblige every rendered bundle to be
    re-rendered whenever a member was added. :func:`outstanding_stage_uris` is
    the scheduled lane's entry point and takes no selection at all -- the
    caller decides whether to ask.
    """

    # Per episode, run only the stages whose steps the catalog does not already
    # record as settled at their current versions.
    OUTSTANDING = "outstanding"
    # Every stage of the profile on every episode, whatever the catalog says.
    # The escape hatch for the one thing the catalog cannot see: an artifact
    # deleted out from under a recorded step (a cleaned `media/` directory
    # still has its `media/contact_sheet` rows).
    EVERY_STAGE = "every-stage"


@dataclass(frozen=True)
class OutstandingStages:
    """A recording with a canonical episode, and which stages still owe work on it.

    ``stages`` empty means every candidate stage's steps are settled -- the
    recording is up to date, which is a different fact from having nothing to
    run it against.
    """

    source_identity: str
    stages: frozenset[Stage]
    # Named so a report can say WHY a stage is scheduled. Empty exactly when
    # `stages` is: one fact, one owner, and the caller never has to reconcile
    # a stage set against a step list that disagrees with it.
    outstanding_steps: tuple[str, ...]


@dataclass(frozen=True)
class NoCanonicalEpisode:
    """Sync produced no episode for this recording, so no later stage can run.

    Deliberately carries no stage set rather than an empty one. The two ways a
    recording ends up running nothing -- up to date, and nothing to run against
    -- are different enough that a report must not add them together, and a
    single type with a ``stages`` field plus a flag would leave "failed sync,
    but here are three stages to run" representable.
    """

    source_identity: str


# Which of the two a recording got is the whole output of planning.
EpisodeStagePlan = OutstandingStages | NoCanonicalEpisode


def _required_steps_by_stage(
    application: "App", candidate_stages: Iterable[Stage]
) -> dict[Stage, tuple[tuple[str, str], ...]]:
    """The ``(name, version)`` pairs each stage is responsible for recording.

    ``sync`` records no step rows -- it produces the canonical episode, and
    that is what its own reuse gate covers -- so it never appears here and is
    never planned away.
    """
    from hflow.app import MEDIA_CONTACT_SHEET_STEP_NAME, media_contact_sheet_step_version

    by_stage: dict[Stage, tuple[tuple[str, str], ...]] = {
        Stage.META: tuple(
            sorted((registered.name, registered.version) for registered in application.checks)
        ),
        Stage.LABELS: tuple(
            sorted((registered.name, registered.version) for registered in application.enrichments)
        ),
        Stage.MEDIA: ((MEDIA_CONTACT_SHEET_STEP_NAME, media_contact_sheet_step_version()),),
    }
    candidates = frozenset(candidate_stages)
    # Iterated in stage-graph order rather than the caller's, so a plan reads
    # the same way twice regardless of what kind of collection it was handed.
    return {stage: by_stage[stage] for stage in Stage if stage in candidates and stage in by_stage}


def _quote_sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _settled_step_names(
    connection: "duckdb.DuckDBPyConnection",
    episode_ids: Sequence[str],
    identity_pairs: Sequence[tuple[str, str]],
) -> dict[str, set[str]]:
    """Per episode id, the step names settled at exactly the versions given.

    Filtering on ``check_version`` inside the query is what lets the result be
    keyed by name alone downstream: a row that came back is a row at the
    current version, so a name present is a step that needs no rerun.

    Deliberately NOT joined on ``run_fingerprint``. A stage-profile run appends
    check_runs rows only for the steps that ran, under a fresh fingerprint, so
    matching this run's fingerprint would report every other stage's work as
    missing after any partial pass -- and the planner would schedule the whole
    graph forever. The pair ``(episode_id, check_version)`` is the identity
    that matters; which run recorded it is provenance.
    """
    if not episode_ids or not identity_pairs:
        return {}
    episode_placeholders = ", ".join("?" for _ in episode_ids)
    identity_clauses = " OR ".join("(check_name = ? AND check_version = ?)" for _ in identity_pairs)
    statuses = ", ".join(_quote_sql_string(status.value) for status in SETTLED_STATUSES)
    parameters: list[str] = list(episode_ids)
    for name, version in identity_pairs:
        parameters.extend((name, version))
    rows = connection.execute(
        f"""
        SELECT episode_id, check_name FROM check_runs
        WHERE status IN ({statuses})
          AND episode_id IN ({episode_placeholders})
          AND ({identity_clauses})
        GROUP BY episode_id, check_name
        """,
        parameters,
    ).fetchall()
    settled: dict[str, set[str]] = {}
    for episode_id, check_name in rows:
        settled.setdefault(str(episode_id), set()).add(str(check_name))
    return settled


def plan_outstanding_stages(
    application: "App",
    source_identities: Sequence[str],
    candidate_stages: Iterable[Stage],
) -> dict[str, EpisodeStagePlan]:
    """Per source recording, which of ``candidate_stages`` still has work to do.

    **Call this only after ``sync`` has run in the same invocation.** Everything
    below reads the catalog as this run's own output, and that is only true then.

    ``source_identities`` are catalog ``source_uri`` values -- ask
    :meth:`hflow.App.source_identity` for one rather than spelling it, or the
    query looks for rows filed under another name and every episode reads as
    outstanding.

    A source with **no** episode in the catalog gets **no** stages, which is the
    post-sync contract doing real work rather than a degenerate case. Sync
    appends an episodes row for every recording it canonicalizes, so after it
    has run, "no row" means it failed on that recording: there is no canonical
    file for a later stage to open, and running one anyway earns a second
    failure that says ``FileNotFoundError`` where the first already said
    ``source-unreadable``. The real failure is counted and classified in
    ``ingest_failures`` by the stage that hit it.

    Known limitation, stated because it costs work rather than correctness: the
    media stage records nothing at all on a camera-less episode (there is no
    contact sheet to render), so "no row" cannot be told from "nothing to
    render" and such an episode is planned for ``media`` on every pass. The
    stage is a no-op for it beyond opening the canonical, and the alternative
    -- treating a missing row as done -- would silently never render a sheet
    for an episode that wanted one.
    """
    from hflow.curation import open_catalog_connection

    required_by_stage = _required_steps_by_stage(application, candidate_stages)
    if not source_identities or not required_by_stage:
        # No stage among the candidates records anything, so every recording is
        # trivially up to date rather than un-runnable.
        return {
            source: OutstandingStages(
                source_identity=source, stages=frozenset(), outstanding_steps=()
            )
            for source in source_identities
        }

    try:
        connection = open_catalog_connection(application.workspace.catalog_root)
    except FileNotFoundError:
        # No catalog at all, so nothing was appended, so sync canonicalized
        # nothing -- the same answer a per-source miss gets, for the same
        # reason. A catalog that exists but cannot be READ (a format-version
        # mismatch, which raises ValueError) is deliberately NOT caught: it
        # would have failed this run's appends already, and swallowing it here
        # would report every stage as up to date when nothing had run.
        return {source: NoCanonicalEpisode(source_identity=source) for source in source_identities}
    try:
        source_placeholders = ", ".join("?" for _ in source_identities)
        episode_by_source = {
            str(source): str(episode_id)
            for source, episode_id in connection.execute(
                f"SELECT source_uri, episode_id FROM episodes_latest "
                f"WHERE source_uri IN ({source_placeholders})",
                list(source_identities),
            ).fetchall()
        }
        identity_pairs = sorted({pair for pairs in required_by_stage.values() for pair in pairs})
        settled_by_episode = _settled_step_names(
            connection, sorted(set(episode_by_source.values())), identity_pairs
        )
    finally:
        connection.close()

    plans: dict[str, EpisodeStagePlan] = {}
    for source in source_identities:
        episode_id = episode_by_source.get(source)
        if episode_id is None:
            plans[source] = NoCanonicalEpisode(source_identity=source)
            continue
        settled = settled_by_episode.get(episode_id, set())
        outstanding_stages: set[Stage] = set()
        outstanding_steps: list[str] = []
        for stage, pairs in required_by_stage.items():
            missing = [name for name, _version in pairs if name not in settled]
            if missing:
                outstanding_stages.add(stage)
                outstanding_steps.extend(missing)
        plans[source] = OutstandingStages(
            source_identity=source,
            stages=frozenset(outstanding_stages),
            outstanding_steps=tuple(sorted(outstanding_steps)),
        )
    return plans


def outstanding_stage_uris(
    application: "App",
    uris: Sequence[str],
    stage: Stage,
    *,
    data_root: str,
) -> list[str]:
    """Which of ``uris`` still owe ``stage`` work, in the order given.

    The same question :func:`plan_outstanding_stages` answers for a direct
    run, asked for one stage and answered in the caller's own vocabulary:
    conf ``uris`` are data-root-relative strings, and the catalog files rows
    under ``source_uri``. Translating between the two is the whole reason this
    is not a bare call at the call site -- computing the identity a second way
    is how a planner ends up querying for rows filed under another name, which
    reads as "everything is outstanding" and quietly does nothing.

    **Call this only after ``sync`` has completed for the same recordings**,
    for the reason in the module docstring: before then the catalog names the
    PREVIOUS generation. The scheduled lane satisfies that by running inside a
    stage sub-DAG, which the master triggers after the sync sub-DAG finished.
    Triggering a stage sub-DAG directly, without a sync pass over recordings
    whose bytes changed, is the one case where this reads a stale id -- the
    same exposure the direct executor has, and what the escape hatch is for.

    ``Stage.SYNC`` is refused rather than answered. It records no step rows and
    is its own cache, so "outstanding" is not a question the catalog can answer
    about it, and an empty answer would filter away the stage that produces the
    canonical file every later stage reads.

    A recording with no episode in the catalog is KEPT, which is the one place
    this deliberately differs from :func:`plan_outstanding_stages`. That
    function drops it, and can, because the direct executor ran sync in the
    same invocation and therefore knows a missing row means sync failed. A
    stage sub-DAG knows no such thing: a directly triggered ``meta`` pass over
    a corpus that was never synced looks identical from the catalog, and
    dropping it would turn the loud ``FileNotFoundError`` that ``hflow ingest
    --profile metadata_backfill`` raises today into a silent skip. Filtering
    exists to stop paying for work that is already done, not to suppress
    failures, so a recording the catalog cannot vouch for is handed on and
    fails exactly as it does now.
    """
    if stage is Stage.SYNC:
        message = "sync records no steps and is never planned away; it cannot be filtered"
        raise ValueError(message)
    if not uris:
        return []
    from hflow.stage_execution import resolve_episode_reference

    identity_by_uri = {
        uri: application.source_identity(resolve_episode_reference(data_root, str(uri)))
        for uri in uris
    }
    plans = plan_outstanding_stages(application, list(identity_by_uri.values()), (stage,))
    return [
        uri
        for uri in uris
        if not isinstance(plan := plans.get(identity_by_uri[uri]), OutstandingStages)
        or stage in plan.stages
    ]
