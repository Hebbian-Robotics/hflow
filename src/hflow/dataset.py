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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from hflow.catalog import EPISODES_VIEW_STATUS_COLUMN
from hflow.curation import CheckCoverage, CurationReport, curate
from hflow.format import EPISODE_FORMAT_VERSION
from hflow.steps import RAN_STATUSES
from hflow.workspace import Workspace

if TYPE_CHECKING:
    from hflow.app import App

MANIFESTS_DIRECTORY_NAME = "manifests"

# Versions this sidecar's shape (not the episode schema, not the catalog
# layout), following identity_version in workspace.json.
DATASET_MANIFEST_VERSION = 1

_FALLBACK_DATASET_SLUG = "dataset"


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


def dataset_slug(raw_name: str) -> str:
    """A user-given name as a filename slug: lowercase, ``[a-z0-9-]``.

    Same rule the workspace server already applies to pinned manifests, so one
    workspace's ``manifests/`` directory keeps one naming convention whichever
    surface wrote the file. A name with no ASCII alphanumerics slugs to the
    fallback rather than being refused.
    """
    slug = "".join(
        character if character.isalnum() and character.isascii() else "-"
        for character in raw_name.lower()
    )
    slug = "-".join(part for part in slug.split("-") if part)
    return slug or _FALLBACK_DATASET_SLUG


def _quote_sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def default_dataset_sql(application: "App") -> str:
    """The SQL for "every episode this pipeline currently stands behind".

    Four rules, each of which excludes something a naive ``SELECT * FROM
    episodes`` would keep:

    1. **Current transform.** ``pipeline_version`` and ``schema_version`` must
       match what this App produces now, so a corpus half-reprocessed after a
       transform change does not mix two canonical behaviors in one dataset.
    2. **Not quarantined.** The pipeline's own critical checks rejected it.
    3. **Every registered check ran, at its current version.** A check added
       last week that has not been backfilled leaves its episodes out rather
       than silently reporting a dataset with a hole in it.
    4. **One row per source recording**, which the ``episodes`` view already
       guarantees, so a reprocessed recording contributes its current
       generation and not both.

    Rule 3 asks whether the check RAN, not whether it passed: the entire
    built-in library is evidence-only and records ``measured``, so a
    ``status = 'passed'`` reading would select nothing at all. See
    :data:`hflow.steps.RAN_STATUSES`.
    """
    ran_statuses = ", ".join(_quote_sql_string(status.value) for status in RAN_STATUSES)
    predicates = [
        f"{EPISODES_VIEW_STATUS_COLUMN} != 'quarantined'",
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
            f"        WHERE status IN ({ran_statuses})\n"
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
    # Microsecond precision, so two datasets of one name in the same second
    # still get distinct files: these are never overwritten.
    file_stem = f"{dataset_slug(name)}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
    manifests_root = workspace.storage_root.child(MANIFESTS_DIRECTORY_NAME)
    effective_sql = sql if sql is not None else default_dataset_sql(application)

    report = curate(
        workspace.catalog_root,
        effective_sql,
        output=manifests_root.uri_for(f"{file_stem}.parquet"),
    )
    sidecar_name = f"{file_stem}.json"
    sidecar_payload = _sidecar_payload(
        application=application,
        name=name,
        sql=effective_sql,
        report=report,
        created_at=stamped_at,
        workspace=workspace,
    )
    manifests_root.write_bytes_if_absent(
        sidecar_name, (json.dumps(sidecar_payload, indent=2, sort_keys=True) + "\n").encode()
    )
    return DatasetManifest(
        manifest_path=report.manifest_path if report.manifest_path is not None else "",
        sidecar_path=manifests_root.uri_for(sidecar_name),
        name=name,
        sql=effective_sql,
        row_count=report.row_count,
        total_episodes=report.total_episodes,
        coverage=report.coverage,
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
