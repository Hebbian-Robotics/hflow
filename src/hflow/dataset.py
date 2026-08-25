"""The default dataset: the corpus a pipeline would stand behind, as a query.

``hflow curate`` takes any SQL, which is the right primitive and the wrong
default. Writing the everyday cut by hand means pasting content-hash versions
into a WHERE clause and remembering four separate rules -- current generation,
current transform, not quarantined, checks actually ran -- and getting any one
of them wrong yields a dataset that looks fine and is not.

So the rules live here, once, and :func:`default_dataset_sql` writes them
down. Nothing is hidden: the emitted SQL is recorded verbatim beside the
manifest, so a dataset can always be explained, audited, and edited into a
sharper one.

The artifact is a pair, both immutable and never overwritten:

- ``manifests/<slug>-<utc timestamp>.parquet`` -- the episode selection, the
  same shape ``hflow curate`` writes and ``hflow export snapshot --manifest``
  reads, and the same layout the workspace server already pins into.
- ``manifests/<slug>-<utc timestamp>.json`` -- what produced it: the SQL, the
  pipeline's version stamps and every step version it required, the row count,
  the coverage, and the workspace it came from.
"""

import json
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from hflow.catalog import EPISODE_STATUS_OK, EPISODES_VIEW_STATUS_COLUMN
from hflow.curation import CheckCoverage, CurationReport, curate
from hflow.format import EPISODE_FORMAT_VERSION
from hflow.steps import SETTLED_STATUSES
from hflow.workspace import MANIFESTS_DIRECTORY_NAME, Workspace

if TYPE_CHECKING:
    from hflow.app import App

# Versions this sidecar's shape (not the episode schema, not the catalog
# layout), following identity_version in workspace.json.
DATASET_MANIFEST_VERSION = 1

_FALLBACK_DATASET_SLUG = "dataset"


class ManifestAlreadyExistsError(FileExistsError):
    """A manifest already occupies that key.

    Never an overwrite: these artifacts are the record of what a dataset was,
    and silently replacing one would make an earlier answer unreproducible
    without anyone noticing. A ``FileExistsError`` so callers that already
    handle one keep working.
    """


@dataclass(frozen=True)
class WrittenManifest:
    """One manifest published into a workspace's ``manifests/``."""

    relative_key: str
    uri: str
    file_stem: str
    report: CurationReport


@dataclass(frozen=True)
class DatasetManifest:
    """One created dataset: where it landed and what it selected."""

    manifest_path: Path | str
    sidecar_path: Path | str
    name: str
    sql: str
    row_count: int
    total_episodes: int
    coverage: list[CheckCoverage]
    created_at: str

    def summary(self) -> str:
        return (
            f"dataset {self.name!r}: {self.row_count} of {self.total_episodes} episodes\n"
            f"  manifest: {self.manifest_path}\n"
            f"  provenance: {self.sidecar_path}"
        )

    def __str__(self) -> str:
        return self.summary()


def dataset_slug(raw_name: str, *, fallback: str = _FALLBACK_DATASET_SLUG) -> str:
    """A user-given name as a filename slug: lowercase, ``[a-z0-9-]``.

    One rule for every surface that writes into ``manifests/``, so a workspace
    keeps one naming convention whichever wrote the file. A name with no ASCII
    alphanumerics slugs to ``fallback`` rather than being refused, because a
    name is a label and refusing one over its alphabet would be absurd.
    """
    slug = "".join(
        character if character.isalnum() and character.isascii() else "-"
        for character in raw_name.lower()
    )
    slug = "-".join(part for part in slug.split("-") if part)
    return slug or fallback


def manifest_file_stem(raw_name: str, *, fallback: str = _FALLBACK_DATASET_SLUG) -> str:
    """``<slug>-<utc timestamp>``, the filename both writers agree on.

    Microsecond precision, so two manifests of one name in the same second
    still get distinct files: nothing here is ever overwritten.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{dataset_slug(raw_name, fallback=fallback)}-{stamp}"


def write_dataset_manifest(
    workspace: Workspace,
    *,
    name: str,
    sql: str,
    constrained: bool = False,
    file_stem: str | None = None,
    fallback_slug: str = _FALLBACK_DATASET_SLUG,
) -> WrittenManifest:
    """Run ``sql`` and publish the result as an immutable manifest.

    The one owner of "write a manifest into this workspace", shared by
    ``hflow dataset create`` and the workspace server's pinning route. Neither
    calls the other: the CLI needs an App for its default policy and provenance
    record, the server has no App to require and must run tenant SQL
    ``constrained``, and both need exactly this.

    Staged into a temporary directory outside the workspace and then published
    create-if-absent, which is what makes "never overwritten" a property of the
    store rather than of a check-then-write race. Bucket roots get the same
    guarantee: the conditional put is the arbiter.
    """
    manifests_root = workspace.manifests_root
    stem = file_stem if file_stem is not None else manifest_file_stem(name, fallback=fallback_slug)
    manifest_key = f"{stem}.parquet"
    if manifests_root.exists(manifest_key):
        # A cheap refusal before the query runs, so a collision never pays for
        # the work. The create-if-absent publish below is the real arbiter.
        raise ManifestAlreadyExistsError(
            f"a manifest already exists at {manifests_root.uri_for(manifest_key)}; "
            "manifests are immutable and never overwritten"
        )
    with tempfile.TemporaryDirectory(prefix="hflow-manifest-") as staging_directory:
        staged_manifest = Path(staging_directory) / manifest_key
        report = curate(
            workspace.catalog_root, sql, output=staged_manifest, constrained=constrained
        )
        if not manifests_root.store_file_if_absent(staged_manifest, manifest_key):
            raise ManifestAlreadyExistsError(
                f"a manifest already exists at {manifests_root.uri_for(manifest_key)}; "
                "manifests are immutable and never overwritten"
            )
    return WrittenManifest(
        relative_key=f"{MANIFESTS_DIRECTORY_NAME}/{manifest_key}",
        uri=manifests_root.uri_for(manifest_key),
        file_stem=stem,
        report=report,
    )


def _quote_sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def default_dataset_sql(application: "App") -> str:
    """The SQL for "every episode this pipeline currently stands behind".

    Four rules, each of which excludes something a naive ``SELECT * FROM
    episodes`` would keep:

    1. **Current transform.** ``pipeline_version`` and ``schema_version`` must
       match what this App produces now, so a corpus half-reprocessed after a
       transform change does not mix two canonical behaviors in one dataset.
    2. **Status is ``ok``.** Excludes two different things. ``quarantined`` is
       the pipeline's own critical checks rejecting the episode. ``unverified``
       is a critical check that crashed, so nobody actually checked it, and
       that is the half rule 3 cannot see: a crash leaves no settled row, but
       an EARLIER settled run of the same check satisfies rule 3 on its own,
       and the episode would otherwise land in the dataset on the strength of
       a result that a later run withdrew.
    3. **Every registered step settled, at its current version.** A check added
       last week that has not been backfilled leaves its episodes out rather
       than silently reporting a dataset with a hole in it.
    4. **One row per source recording**, which the ``episodes`` view already
       guarantees, so a reprocessed recording contributes its current
       generation and not both.

    Rules 2 and 3 overlap without either being redundant. An episode whose
    critical check ONLY ever crashed is dropped by rule 3, which needs a
    settled row and never gets one, and rule 2 agrees. An episode that settled
    once and crashed later is dropped by rule 2 alone. Neither rule subsumes
    the other, so both stay.

    Rule 3 has two traps in it, and both of them yield an EMPTY dataset that
    looks like a policy decision:

    - It asks whether the step settled, never whether it PASSED. The entire
      built-in library is evidence-only and records ``measured``, so a
      ``status = 'passed'`` reading selects nothing at all.
    - "Settled" is wider than "ran", because a default check the pipeline
      supersedes records ``skipped`` on every episode forever -- and wrapping
      a built-in to configure it is the documented way to configure one, so
      reading ``skipped`` as an unfilled hole empties the dataset of the
      pipelines most likely to want it. See :data:`hflow.steps.SETTLED_STATUSES`
      for why that is safe here and where the two differ.
    """
    settled_statuses = ", ".join(_quote_sql_string(status.value) for status in SETTLED_STATUSES)
    predicates = [
        f"{EPISODES_VIEW_STATUS_COLUMN} = '{EPISODE_STATUS_OK}'",
        f"pipeline_version = {_quote_sql_string(application.pipeline_version)}",
        f"schema_version = {_quote_sql_string(EPISODE_FORMAT_VERSION)}",
    ]
    required_steps = sorted(
        {(registered.name, registered.version) for registered in application.checks}
        | {(registered.name, registered.version) for registered in application.enrichments}
    )
    if required_steps:
        identity_clauses = " OR ".join(
            f"(check_name = {_quote_sql_string(name)} "
            f"AND check_version = {_quote_sql_string(version)})"
            for name, version in required_steps
        )
        predicates.append(
            "episode_id IN (\n"
            "        SELECT episode_id FROM check_runs\n"
            f"        WHERE status IN ({settled_statuses})\n"
            f"          AND ({identity_clauses})\n"
            "        GROUP BY episode_id\n"
            f"        HAVING count(DISTINCT check_name) = {len(required_steps)}\n"
            "    )"
        )
    joined_predicates = "\n  AND ".join(predicates)
    return f"SELECT *\nFROM episodes\nWHERE {joined_predicates}\nORDER BY episode_id"


def create_dataset(
    application: "App",
    name: str,
    *,
    sql: str | None = None,
    created_at: str | None = None,
) -> DatasetManifest:
    """Write one immutable dataset manifest and its provenance sidecar.

    ``sql`` overrides the default policy for callers that want the artifact
    and the provenance record without the opinion; whatever runs is what the
    sidecar records, so the two can never disagree.
    """
    workspace = Workspace(application.storage_root)
    stamped_at = created_at if created_at is not None else datetime.now(UTC).isoformat()
    effective_sql = sql if sql is not None else default_dataset_sql(application)

    written = write_dataset_manifest(workspace, name=name, sql=effective_sql)
    sidecar_name = f"{written.file_stem}.json"
    sidecar_payload = _sidecar_payload(
        application=application,
        name=name,
        sql=effective_sql,
        report=written.report,
        created_at=stamped_at,
        workspace=workspace,
    )
    # The pair is two publishes and cannot be made one, so the failure worth
    # ruling out is the SILENT one: a manifest that landed beside a provenance
    # record that did not, reported as a complete dataset. Manifests are never
    # overwritten, so the same name cannot be retried into the same stem --
    # naming both files is what makes the half-written pair recoverable by
    # hand. store_file_if_absent returning False means the key was taken
    # between the manifest publish and this one, which is a collision on a
    # microsecond-stamped stem and means someone is racing this command.
    if not workspace.manifests_root.write_bytes_if_absent(
        sidecar_name, (json.dumps(sidecar_payload, indent=2, sort_keys=True) + "\n").encode()
    ):
        raise ManifestAlreadyExistsError(
            f"the manifest {written.uri} was written, but its provenance record "
            f"{workspace.manifests_root.uri_for(sidecar_name)} already existed, so "
            "the pair is incomplete; the manifest is still valid and readable, and "
            "`hflow dataset create` under another name will produce a complete pair"
        )
    return DatasetManifest(
        manifest_path=written.uri,
        sidecar_path=workspace.manifests_root.uri_for(sidecar_name),
        name=name,
        sql=effective_sql,
        row_count=written.report.row_count,
        total_episodes=written.report.total_episodes,
        coverage=written.report.coverage,
        created_at=stamped_at,
    )


def _sidecar_payload(
    *,
    application: "App",
    name: str,
    sql: str,
    report: CurationReport,
    created_at: str,
    workspace: Workspace,
) -> dict[str, object]:
    """Everything needed to explain this dataset without the pipeline in hand."""
    manifest = application.manifest()
    workspace_identity = workspace.identity()
    return {
        "dataset_manifest_version": DATASET_MANIFEST_VERSION,
        "name": name,
        "created_at": created_at,
        "sql": sql,
        "row_count": report.row_count,
        "total_episodes": report.total_episodes,
        "coverage": [
            {
                "check_name": entry.check_name,
                "episodes_ran": entry.episodes_ran,
                "total_episodes": entry.total_episodes,
            }
            for entry in sorted(report.coverage, key=lambda entry: entry.check_name)
        ],
        "pipeline": json.loads(manifest.to_json()),
        "workspace_id": workspace_identity.workspace_id if workspace_identity else None,
    }
