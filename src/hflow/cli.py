"""Command-line entry point.

Subcommands: ``curate``, ``dataset create``, ``export snapshot``, ``stale``,
``doctor``, ``manifest``, the Compose runtime family
``up``/``down``/``ingest``/``status``, ``deploy`` for bring-your-own Airflow,
and ``serve`` for the workspace HTTP server (a separate ``hflow-server``
package, imported only when invoked).
Everything the CLI does is a thin call into the library: no behavior lives
only here.

Two of these start long-running processes and they are not the same thing:
``up`` brings up the RUNTIME that processes episodes (an Airflow stack in
Docker), while ``serve`` serves the WORKSPACE over HTTP -- one process that
reads the data root and can trigger a run on a runtime, but executes nothing
itself. Either is useful without the other.

``ingest`` and ``status`` address either a LOCAL rendered bundle (the
default: ``--bundle-dir`` or its auto-discovery) or a REMOTE runtime by URL
(``--airflow-url`` / ``HFLOW_AIRFLOW_URL`` plus ``HFLOW_AIRFLOW_DAG_ID`` and
environment credentials) -- the same commands drive a hosted workspace.
"""

import argparse
import errno
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from hflow import __version__
from hflow.app import (
    DEFAULT_DATA_ROOT,
    default_data_root,
    resolve_pipeline_spec_for_rendering,
)
from hflow.curation import curate, stale_episodes
from hflow.doctor import diagnose
from hflow.project import (
    DEFAULT_PIPELINE_FILE_NAME,
    PROJECT_CONFIG_FILE_NAME,
    ProjectConfig,
    find_project_config,
)
from hflow.runtime._deploy import DEFAULT_DEPLOY_VENV_PYTHON
from hflow.steps import RUN_PROFILES
from hflow.storage import is_bucket_url
from hflow.workspace import CATALOG_DIRECTORY_NAME, RUNTIME_BUNDLE_DIRECTORY_NAME

if TYPE_CHECKING:
    from hflow.app import App
    from hflow.runtime import RemoteRuntimeEndpoint

DEFAULT_DEPLOY_OUTPUT_DIR = Path("./deploy")
# Mirrors RuntimeConfig.api_port; kept here so the parser can state it without
# importing the runtime package, which `up` defers until it actually runs.
DEFAULT_API_PORT = 8080
# The workspace UI's fixed default port ("HFLO" on a phone keypad); stated
# here so the parser needs no import from the optional hflow-server package.
DEFAULT_SERVER_PORT = 4356


def _environment_data_root() -> str:
    """The data root these commands default to when no flag names one.

    Literally :func:`hflow.app.default_data_root`, as a string for the URL and
    path joins below. Shared rather than reimplemented: an App written as
    ``hflow.App("name")`` resolves its root through the same function, so
    ``hflow ingest`` writes the workspace ``hflow curate`` reads.
    """
    return str(default_data_root())


def _default_catalog_location() -> str:
    # A string join, not Path: bucket URLs (gs://...) must survive.
    return f"{_environment_data_root().rstrip('/')}/{CATALOG_DIRECTORY_NAME}"


def _configured_pipeline_spec() -> str | None:
    """The pipeline this project points at, if it points at one.

    ``hflow.toml``'s ``pipeline``, else ``pipeline.py`` beside it. ``None`` is
    an ordinary answer, not a failure: ``hflow serve`` runs fine without a
    pipeline and simply turns its pipeline page off.

    The conventional fallback looks in the PROJECT's directory, not the
    working one, whenever an ``hflow.toml`` located the project. Otherwise
    running a command from ``notebooks/`` would resolve the data root from the
    project and the pipeline from wherever the shell happened to be, which is
    the one combination guaranteed to address two different things.
    """
    project_config = find_project_config()
    if isinstance(project_config, ProjectConfig):
        if project_config.pipeline_file is not None:
            return str(project_config.pipeline_file)
        search_directory = project_config.config_file.parent
    else:
        search_directory = Path.cwd()
    conventional_pipeline = search_directory / DEFAULT_PIPELINE_FILE_NAME
    return str(conventional_pipeline) if conventional_pipeline.is_file() else None


def _require_pipeline_spec(explicit_pipeline: str | None) -> str:
    """The pipeline address for a command that cannot run without one."""
    if explicit_pipeline is not None:
        return explicit_pipeline
    configured_pipeline = _configured_pipeline_spec()
    if configured_pipeline is None:
        raise ValueError(
            "no pipeline found: pass --pipeline path/to/pipeline.py, add "
            f'`pipeline = "..."` to {PROJECT_CONFIG_FILE_NAME}, or run from a '
            f"directory holding {DEFAULT_PIPELINE_FILE_NAME}"
        )
    return configured_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hflow",
        description="Open-source robotics data pipeline.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"hflow {__version__}",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    curate_parser = subparsers.add_parser(
        "curate",
        help="run a SQL query over the episode catalog and write manifest.parquet",
        description=(
            "Run any SELECT over the catalog views (the wide 'episodes' view "
            "covers everyday cuts) and write the result as a Parquet manifest."
        ),
    )
    curate_parser.add_argument(
        "sql",
        nargs="?",
        help="the SELECT to run (or pass --sql-file)",
    )
    curate_parser.add_argument(
        "--sql-file",
        type=Path,
        help="read the SELECT from a file instead of the command line",
    )
    curate_parser.add_argument(
        "--catalog",
        default=_default_catalog_location(),
        help=(
            "catalog directory or object-store prefix "
            f"(default: $HFLOW_DATA_ROOT, else {PROJECT_CONFIG_FILE_NAME}'s data_root, "
            f"else {DEFAULT_DATA_ROOT} -- plus /catalog)"
        ),
    )
    curate_output_group = curate_parser.add_mutually_exclusive_group()
    curate_output_group.add_argument(
        "--output",
        "-o",
        default=f"{_environment_data_root().rstrip('/')}/manifest.parquet",
        help=(
            "manifest path or object-store URL "
            f"(default: $HFLOW_DATA_ROOT, else {PROJECT_CONFIG_FILE_NAME}'s data_root, "
            f"else {DEFAULT_DATA_ROOT} -- plus /manifest.parquet)"
        ),
    )
    curate_output_group.add_argument(
        "--dry-run",
        action="store_true",
        help="run the query and report row count and coverage without writing a manifest",
    )

    dataset_parser = subparsers.add_parser(
        "dataset",
        help="create version-pinned dataset manifests from the pipeline's own policy",
    )
    dataset_subparsers = dataset_parser.add_subparsers(dest="dataset_command", required=True)
    dataset_create_parser = dataset_subparsers.add_parser(
        "create",
        help="write an immutable manifest of every episode this pipeline stands behind",
        description=(
            "Select the current generation of every source recording that is not "
            "quarantined, was produced by this pipeline's current transform, and "
            "has every registered step recorded at its current version. Writes "
            "manifests/<name>-<timestamp>.parquet plus a .json recording the "
            "effective SQL and the versions it required. Importing EXECUTES the "
            "pipeline file, so run this in the pipeline's own environment."
        ),
    )
    dataset_create_parser.add_argument(
        "name",
        help="a name for the dataset; slugified into the manifest's filename",
    )
    dataset_create_parser.add_argument(
        "--pipeline",
        default=None,
        help=(
            "pipeline file, optionally with the App variable name: "
            "path/to/pipeline.py[:app]. Defaults to hflow.toml's `pipeline`, "
            f"else ./{DEFAULT_PIPELINE_FILE_NAME}"
        ),
    )
    dataset_create_parser.add_argument(
        "--sql",
        default=None,
        help=(
            "replace the default policy with your own SELECT, keeping the "
            "immutable artifact and the provenance record"
        ),
    )
    dataset_create_parser.add_argument(
        "--print-sql",
        action="store_true",
        help="print the SQL this would run and exit, writing nothing",
    )

    export_parser = subparsers.add_parser(
        "export",
        help="export catalog selections in portable downstream formats",
    )
    export_subparsers = export_parser.add_subparsers(dest="export_command", required=True)
    snapshot_export_parser = export_subparsers.add_parser(
        "snapshot",
        help="write a tool-neutral Parquet dataset snapshot",
        description=(
            "Snapshot selected episodes, measurements, artifact media, check runs, "
            "tags, and intervals into a local directory of standard Parquet files."
        ),
    )
    snapshot_export_parser.add_argument(
        "--catalog",
        default=_default_catalog_location(),
        help=(
            "catalog directory or object-store prefix "
            f"(default: $HFLOW_DATA_ROOT, else {PROJECT_CONFIG_FILE_NAME}'s data_root, "
            f"else {DEFAULT_DATA_ROOT} -- plus /catalog)"
        ),
    )
    snapshot_export_parser.add_argument(
        "--manifest",
        default=None,
        help=(
            "optional local Parquet file or object-store URL containing episode_id; "
            "without it, export every latest catalog episode"
        ),
    )
    snapshot_export_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="local directory to create",
    )
    snapshot_export_parser.add_argument(
        "--media",
        choices=("references", "copy"),
        default="references",
        help=(
            "preserve artifact URIs, or copy artifacts under the export's assets/ "
            "directory (default: references)"
        ),
    )
    snapshot_export_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="atomically replace an existing export directory",
    )

    stale_parser = subparsers.add_parser(
        "stale",
        help="list episodes whose latest cataloged run predates the current pipeline version",
        description=(
            "Print the source URI of every episode whose latest cataloged run was "
            "produced by a different pipeline (and format) version -- one per line "
            "on stdout, ready to pipe back into `hflow ingest` for selective "
            "reprocessing. The summary goes to stderr."
        ),
    )
    stale_parser.add_argument(
        "--catalog",
        default=_default_catalog_location(),
        help=(
            "catalog directory or object-store prefix "
            f"(default: $HFLOW_DATA_ROOT, else {PROJECT_CONFIG_FILE_NAME}'s data_root, "
            f"else {DEFAULT_DATA_ROOT} -- plus /catalog)"
        ),
    )
    # Not required: without either flag the pipeline is resolved from
    # hflow.toml or ./pipeline.py, exactly as the other commands do.
    stale_group = stale_parser.add_mutually_exclusive_group()
    stale_group.add_argument(
        "--pipeline",
        help=(
            "pipeline file to compute the current version from, optionally with the "
            "App variable name: path/to/pipeline.py[:app]. Defaults to hflow.toml's "
            f"`pipeline`, else ./{DEFAULT_PIPELINE_FILE_NAME}"
        ),
    )
    stale_group.add_argument(
        "--pipeline-version",
        help="compare against this pipeline_version hash directly (no pipeline import)",
    )
    stale_parser.add_argument(
        "--exit-code",
        action="store_true",
        help=(
            "exit 1 when at least one stale episode is found (like "
            "`git diff --exit-code`), so CI can gate on it; without this flag the "
            "command always exits 0"
        ),
    )

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="check a file against the canonical-episode convention",
        description=(
            "Validate container integrity, metadata stamps, chunk-group layout, "
            "and in-band video constraints (docs/FORMAT.md, executable form). "
            "Accepts multiple files, each reported in order whatever happens "
            "to the others. Exit 0 when all conform, 1 when any file is "
            "non-conforming or could not be read, 2 when no file could be "
            "diagnosed (nothing useful happened)."
        ),
    )
    doctor_parser.add_argument(
        "file", nargs="+", help="the local paths or object-store URLs to check"
    )

    manifest_parser = subparsers.add_parser(
        "manifest",
        help="print the pipeline's manifest (steps, versions, endpoints) as JSON",
        description=(
            "Import the pipeline file and print its manifest -- step names, "
            "content-hash versions, gate flags, endpoint aliases, and version "
            "stamps -- as JSON on stdout. This is the metadata a pipeline "
            "crosses a control boundary as. Importing EXECUTES the pipeline "
            "file, so run this in the pipeline's own environment."
        ),
    )
    manifest_parser.add_argument(
        "--pipeline",
        default=None,
        help=(
            "pipeline file, optionally with the App variable name: "
            "path/to/pipeline.py[:app]. Defaults to hflow.toml's `pipeline`, "
            f"else ./{DEFAULT_PIPELINE_FILE_NAME}"
        ),
    )

    up_parser = subparsers.add_parser(
        "up",
        help="render the Compose bundle and start the local Airflow runtime",
        description=(
            "Render a self-contained Docker Compose bundle for the pipeline, start it "
            "detached, wait until Airflow reports healthy, and print how to reach it. "
            "The first start pulls images (minutes) and builds the user venv."
        ),
    )
    up_parser.add_argument(
        "--pipeline",
        default=None,
        help=(
            "pipeline file, optionally with the App variable name: "
            "path/to/pipeline.py[:app]. Defaults to hflow.toml's `pipeline`, "
            f"else ./{DEFAULT_PIPELINE_FILE_NAME}"
        ),
    )
    up_parser.add_argument(
        "--data-root",
        type=str,
        default=_environment_data_root(),
        help=(
            "host directory mounted at /opt/airflow/data, or a bucket URL "
            "(gs://, s3://, az://) the runtime talks to natively "
            f"(default: $HFLOW_DATA_ROOT, else {PROJECT_CONFIG_FILE_NAME}'s "
            f"data_root, else {DEFAULT_DATA_ROOT})"
        ),
    )
    up_parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=None,
        help="where to render the bundle (default: <data-root>/runtime; ./runtime for bucket URLs)",
    )
    up_parser.add_argument(
        "--api-port",
        type=int,
        default=DEFAULT_API_PORT,
        help=(
            "host port for the Airflow API, written into a new bundle's .env as "
            f"API_PORT (default: {DEFAULT_API_PORT}, range 1-65535). An existing "
            ".env is never rewritten, so this only takes effect on a bundle that "
            "does not have one yet"
        ),
    )
    up_parser.add_argument(
        "--hflow-source",
        type=Path,
        default=None,
        help=(
            "development source checkout to install into the user venv "
            "(default: inferred for editable installs; otherwise the current published version)"
        ),
    )
    up_parser.add_argument(
        "--requirements",
        type=Path,
        default=None,
        help="user requirements file for the task venv (default: hflow only)",
    )

    deploy_parser = subparsers.add_parser(
        "deploy",
        help="emit the DAG bundle for an existing Airflow 3 deployment",
        description=(
            "Render the ingest DAG, the user/ files, and a DEPLOY.md with concrete "
            "placement instructions for Astronomer, MWAA, Cloud Composer, and "
            "self-managed Airflow 3. Emits plain files only -- no platform API is called."
        ),
    )
    deploy_parser.add_argument(
        "--pipeline",
        default=None,
        help=(
            "pipeline file, optionally with the App variable name: "
            "path/to/pipeline.py[:app]. Defaults to hflow.toml's `pipeline`, "
            f"else ./{DEFAULT_PIPELINE_FILE_NAME}"
        ),
    )
    deploy_parser.add_argument(
        "--data-root-uri",
        required=True,
        help=("absolute filesystem path or object-store prefix where episode URIs resolve"),
    )
    deploy_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DEPLOY_OUTPUT_DIR,
        help=f"where to write the bundle (default: {DEFAULT_DEPLOY_OUTPUT_DIR})",
    )
    deploy_parser.add_argument(
        "--requirements",
        type=Path,
        default=None,
        help="user requirements file for the task venv (default: hflow only)",
    )
    deploy_parser.add_argument(
        "--venv-python",
        default=DEFAULT_DEPLOY_VENV_PYTHON,
        help=(
            "the user venv's python interpreter on the workers "
            f"(default: {DEFAULT_DEPLOY_VENV_PYTHON})"
        ),
    )

    down_parser = subparsers.add_parser(
        "down",
        help="stop the Compose runtime (containers only; volumes survive)",
    )
    down_parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=None,
        help=(
            "the rendered bundle to stop (default: $HFLOW_DATA_ROOT/runtime, "
            f"else {DEFAULT_DATA_ROOT}/runtime; ./runtime for bucket data roots)"
        ),
    )
    down_parser.add_argument(
        "--volumes",
        action="store_true",
        help="also remove the metadata-DB and user-venv volumes (full reset)",
    )

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="trigger the master ingest DAG over episode URIs (relative to the data root)",
    )
    ingest_parser.add_argument(
        "uris",
        nargs="+",
        help="episode files, relative to the configured data root",
    )
    ingest_parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=None,
        help=(
            "the rendered bundle to talk to (default: $HFLOW_DATA_ROOT/runtime, "
            f"else {DEFAULT_DATA_ROOT}/runtime; ./runtime for bucket data roots)"
        ),
    )
    _add_remote_endpoint_arguments(ingest_parser)
    ingest_parser.add_argument(
        "--profile",
        choices=sorted(RUN_PROFILES),
        default="full",
        help="run profile: which stage sub-DAGs the master enables (default: full)",
    )
    ingest_parser.add_argument(
        "--all-stages",
        action="store_true",
        help=(
            "run every stage of --profile on every episode, instead of only the "
            "stages whose steps the catalog does not already record at their "
            "current versions. Use it when an artifact was deleted out from "
            "under a recorded step -- that is the one thing the catalog cannot "
            "see. Applies only when the episodes are processed in this process"
        ),
    )
    ingest_parser.add_argument(
        "--online",
        action="store_true",
        help=(
            "latency-first online lane: process the URIs as one immediate batch "
            "(no bin-packing, no stagger) -- for per-episode runs as data lands"
        ),
    )
    ingest_parser.add_argument(
        "--pipeline",
        default=None,
        help=(
            "pipeline to run when no runtime is addressed and the episodes are "
            "processed in this process; ignored when a bundle or --airflow-url "
            f"is addressed, since the runtime holds its own copy. Defaults to "
            f"{PROJECT_CONFIG_FILE_NAME}'s `pipeline`, else ./{DEFAULT_PIPELINE_FILE_NAME}"
        ),
    )

    status_parser = subparsers.add_parser(
        "status",
        help="runtime health summary with plain-language diagnostics",
    )
    status_parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=None,
        help=(
            "the rendered bundle to inspect (default: $HFLOW_DATA_ROOT/runtime, "
            f"else {DEFAULT_DATA_ROOT}/runtime; ./runtime for bucket data roots)"
        ),
    )
    _add_remote_endpoint_arguments(status_parser)

    serve_parser = subparsers.add_parser(
        "serve",
        help=(
            "serve this workspace over HTTP: a REST API over the catalog, and any "
            "UI assets installed (requires the hflow-server package). Distinct from "
            "`up`, which starts the runtime that PROCESSES episodes -- this only "
            "reads the data root, and can trigger a run on a runtime that exists."
        ),
    )
    serve_parser.add_argument(
        "--data-root",
        default=_environment_data_root(),
        help=(
            f"workspace data root to browse (default: $HFLOW_DATA_ROOT, else {DEFAULT_DATA_ROOT})"
        ),
    )
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind address (default 127.0.0.1; widening past loopback exposes your corpus)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_SERVER_PORT,
        help=f"port to serve on (default {DEFAULT_SERVER_PORT}; auto-retries upward when taken)",
    )
    serve_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="do not open a browser after starting (headless use)",
    )
    serve_parser.add_argument(
        "--read-only",
        action="store_true",
        help="refuse manifest pinning, saved-query edits, and run triggering",
    )
    serve_parser.add_argument(
        "--pipeline",
        default=None,
        help=(
            "pipeline file for the Pipeline page, optionally with the App variable "
            "name: path/to/pipeline.py[:app]; importing EXECUTES the file, exactly "
            "like `hflow manifest`"
        ),
    )
    return parser


def _add_remote_endpoint_arguments(command_parser: argparse.ArgumentParser) -> None:
    """The remote-runtime addressing flags ``ingest`` and ``status`` share."""
    command_parser.add_argument(
        "--airflow-url",
        default=None,
        help=(
            "Airflow API base URL of a remote runtime, e.g. a hosted workspace "
            "(or export HFLOW_AIRFLOW_URL); credentials come from the environment "
            "only: HFLOW_AIRFLOW_TOKEN, or HFLOW_AIRFLOW_USERNAME and "
            "HFLOW_AIRFLOW_PASSWORD"
        ),
    )
    command_parser.add_argument(
        "--dag-id",
        default=None,
        help="master ingest DAG id on the remote runtime (or export HFLOW_AIRFLOW_DAG_ID)",
    )


def _remote_endpoint_for_command(arguments: argparse.Namespace) -> "RemoteRuntimeEndpoint | None":
    """The remote endpoint this command addresses, or ``None`` for local.

    An explicit ``--bundle-dir`` keeps the command local even when
    ``HFLOW_AIRFLOW_URL`` is exported; an explicit ``--airflow-url`` wins the
    other way. Raises ``ValueError`` when a remote resolution is incomplete.
    """
    from hflow.runtime import resolve_remote_endpoint

    if arguments.airflow_url is None and arguments.bundle_dir is not None:
        return None
    return resolve_remote_endpoint(airflow_url=arguments.airflow_url, dag_id=arguments.dag_id)


def _found_bundle_dir(bundle_dir_argument: Path | None) -> Path | None:
    """The bundle this command addresses, or ``None`` if there is none.

    ``hflow.runtime.find_bundle_directory`` owns the probe, so the CLI and the
    workspace server cannot disagree about which runtime a workspace has.
    """
    from hflow.runtime import find_bundle_directory

    if bundle_dir_argument is not None:
        return bundle_dir_argument
    return find_bundle_directory(_environment_data_root())


def _resolve_bundle_dir(bundle_dir_argument: Path | None) -> Path:
    """The bundle a command addresses, naming a candidate even when absent.

    For the commands that cannot proceed without one (``down``, ``status``):
    falling back to the primary candidate is what makes ``load_bundle``'s
    error name a path the user recognizes rather than reporting nothing.
    """
    found_bundle_dir = _found_bundle_dir(bundle_dir_argument)
    if found_bundle_dir is not None:
        return found_bundle_dir
    environment_data_root = _environment_data_root()
    if is_bucket_url(environment_data_root):
        return Path(RUNTIME_BUNDLE_DIRECTORY_NAME)
    return Path(environment_data_root) / RUNTIME_BUNDLE_DIRECTORY_NAME


def _import_pipeline_app(pipeline_spec: str) -> "App":
    """Import ``path/to/pipeline.py[:app]`` and return its App, loudly.

    The library owns the contract (:func:`hflow.app.import_pipeline_application`)
    so every vantage that addresses a pipeline by file -- these commands and
    the workspace UI -- resolves it identically.
    """
    from hflow.app import import_pipeline_application

    return import_pipeline_application(pipeline_spec)


def _command_dataset_create(arguments: argparse.Namespace) -> int:
    from hflow.dataset import create_dataset, default_dataset_sql

    try:
        app = _import_pipeline_app(_require_pipeline_spec(arguments.pipeline))
    except ValueError as error:
        print(f"dataset create: {error}", file=sys.stderr)
        return 2
    if arguments.print_sql:
        # The policy is never hidden: it can always be read, edited, and
        # handed back through --sql or `hflow curate`.
        print(arguments.sql if arguments.sql is not None else default_dataset_sql(app))
        return 0
    try:
        dataset = create_dataset(app, arguments.name, sql=arguments.sql)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        print(f"dataset create: {error}", file=sys.stderr)
        return 2
    print(dataset.summary())
    return 0


def _command_manifest(arguments: argparse.Namespace) -> int:
    try:
        app = _import_pipeline_app(_require_pipeline_spec(arguments.pipeline))
    except ValueError as error:
        print(f"manifest: {error}", file=sys.stderr)
        return 2
    sys.stdout.write(app.manifest().to_json())
    return 0


def _command_stale(arguments: argparse.Namespace) -> int:
    schema_version: str | None = None
    if arguments.pipeline_version is None:
        from hflow.format import EPISODE_FORMAT_VERSION

        try:
            app = _import_pipeline_app(_require_pipeline_spec(arguments.pipeline))
        except ValueError as error:
            print(f"stale: {error}", file=sys.stderr)
            return 2
        pipeline_version = app.pipeline_version
        # A pipeline defines the whole current target, format version included.
        schema_version = EPISODE_FORMAT_VERSION
    else:
        pipeline_version = arguments.pipeline_version

    try:
        stale = stale_episodes(
            arguments.catalog,
            pipeline_version=pipeline_version,
            schema_version=schema_version,
        )
    except (ValueError, FileNotFoundError) as error:
        print(f"stale: {error}", file=sys.stderr)
        return 2
    for episode in stale:
        print(episode.source_uri if episode.source_uri is not None else episode.uri)
    print(
        f"stale: {len(stale)} episode(s) behind pipeline_version {pipeline_version}"
        + (f" / schema_version {schema_version}" if schema_version is not None else ""),
        file=sys.stderr,
    )
    if arguments.exit_code and stale:
        return 1
    return 0


def _command_up(arguments: argparse.Namespace) -> int:
    from hflow.runtime import (
        RuntimeConfig,
        infer_hflow_source,
        start_runtime,
        started_summary,
    )

    try:
        pipeline_file, app_variable = resolve_pipeline_spec_for_rendering(
            _require_pipeline_spec(arguments.pipeline)
        )
    except ValueError as error:
        print(f"up: {error}", file=sys.stderr)
        return 2
    hflow_source = (
        arguments.hflow_source if arguments.hflow_source is not None else infer_hflow_source()
    )
    try:
        config = RuntimeConfig(
            pipeline_file=pipeline_file,
            data_root=arguments.data_root,
            app_variable=app_variable,
            requirements_file=arguments.requirements,
            hflow_source=hflow_source,
            api_port=arguments.api_port,
        )
    except ValueError as error:
        # Its own block, not the start_runtime handler below: nothing has been
        # rendered or started yet, so that handler's teardown advice would all
        # be false.
        print(f"up: {error}", file=sys.stderr)
        return 2
    # A bucket data root has no local directory to check, the same distinction
    # drawn for the bundle dir below. Only a root that exists and is not a
    # directory is refused here: every one of the three `mkdir` calls in
    # render_bundle raises NotADirectoryError against it, and nothing has been
    # rendered or started when that happens, so it is bad input (2).
    # A *missing* local root is deliberately left alone. `serve` refuses that
    # case too, but whether it should is open on #143, and until that lands the
    # two commands agreeing on the wrong answer is worse than only this one
    # answering the case that is bad input under any reading.
    if not is_bucket_url(arguments.data_root):
        local_data_root = Path(arguments.data_root)
        if local_data_root.exists() and not local_data_root.is_dir():
            print(
                f"up: {os.strerror(errno.ENOTDIR)}: {local_data_root}",
                file=sys.stderr,
            )
            return 2
    # A bucket data root has no local directory to host the bundle: ./runtime.
    default_bundle_dir = (
        Path(RUNTIME_BUNDLE_DIRECTORY_NAME)
        if is_bucket_url(arguments.data_root)
        else Path(arguments.data_root) / RUNTIME_BUNDLE_DIRECTORY_NAME
    )
    bundle_dir = arguments.bundle_dir if arguments.bundle_dir is not None else default_bundle_dir
    from hflow.runtime import ComposeError

    # Narration goes to stderr so stdout stays exactly the final summary
    # (scripts can capture it; humans see progress on a slow first start).
    def print_progress_to_stderr(message: str) -> None:
        print(f"up: {message}", file=sys.stderr)

    try:
        paths, _ = start_runtime(config, bundle_dir, on_progress=print_progress_to_stderr)
    except FileNotFoundError as error:
        # Its own block, not the tuple below. This is raised while rendering the
        # bundle, before any container exists, so the teardown advice attached to
        # that handler would send the caller after containers that were never
        # created. Same reason it exits 2 (bad input, nothing started) and not 1
        # (started, then failed).
        print(f"up: {error}", file=sys.stderr)
        return 2
    except (ComposeError, TimeoutError) as error:
        # Deliberately leave whatever started running: the state is the
        # diagnosis. Tell the user how to look at it and how to tear it down.
        print(f"up: {error}", file=sys.stderr)
        print(
            "\n".join(
                [
                    f"containers may still be running for the bundle at {bundle_dir}.",
                    f"  inspect:   hflow status --bundle-dir {bundle_dir}",
                    f"  logs:      docker compose --file {bundle_dir}/docker-compose.yaml logs <service>",
                    f"  tear down: hflow down --bundle-dir {bundle_dir}",
                ]
            ),
            file=sys.stderr,
        )
        return 1
    print(started_summary(paths))
    return 0


def _command_deploy(arguments: argparse.Namespace) -> int:
    from hflow.runtime._deploy import DeployConfig, render_deploy_bundle

    try:
        pipeline_file, app_variable = resolve_pipeline_spec_for_rendering(
            _require_pipeline_spec(arguments.pipeline)
        )
    except ValueError as error:
        print(f"deploy: {error}", file=sys.stderr)
        return 2
    try:
        config = DeployConfig(
            pipeline_file=pipeline_file,
            data_root_uri=arguments.data_root_uri,
            app_variable=app_variable,
            requirements_file=arguments.requirements,
            venv_python_path=arguments.venv_python,
        )
        paths = render_deploy_bundle(config, arguments.output_dir)
    except (ValueError, FileNotFoundError) as error:
        print(f"deploy: {error}", file=sys.stderr)
        return 2
    print(
        "\n".join(
            [
                f"deploy bundle: {paths.output_dir}",
                f"ingest DAG:    {paths.dag_file} (dag_id: {paths.dag_id})",
                f"user files:    {paths.user_dir}",
                f"next steps:    read {paths.deploy_md} -- placement per platform, the "
                "task venv, and the environment the DAG expects",
            ]
        )
    )
    return 0


def _command_down(arguments: argparse.Namespace) -> int:
    from hflow.runtime import compose_down, load_bundle

    try:
        paths = load_bundle(_resolve_bundle_dir(arguments.bundle_dir))
    except (ValueError, FileNotFoundError) as error:
        print(f"down: {error}", file=sys.stderr)
        return 2
    compose_down(paths.compose_file, remove_volumes=arguments.volumes)
    print(
        f"runtime at {paths.bundle_dir} stopped{' (volumes removed)' if arguments.volumes else ''}"
    )
    return 0


def _ingest_in_process(arguments: argparse.Namespace) -> int:
    """Ingest with no runtime at all: import the pipeline and run the stages.

    The third executor. A workspace with no rendered bundle and no
    ``HFLOW_AIRFLOW_URL`` used to be a failure (``run `hflow up` first``);
    it is now an ordinary case, because the scale that needs a scheduler and
    the scale that needs one command are different scales.

    ``--online`` and ``--bundle-dir`` have nothing to answer here: there is
    one process, so there are no lanes to pick between and no bundle to
    address. ``--profile`` still selects which stages run, and ``--all-stages``
    turns off the per-episode planning that would otherwise skip the ones the
    catalog already records as current.
    """
    from hflow.stage_execution import run_stages_directly
    from hflow.stage_planning import StageSelection
    from hflow.steps import stages_for_profile

    try:
        app = _import_pipeline_app(_require_pipeline_spec(arguments.pipeline))
    except ValueError as error:
        print(f"ingest: {error}", file=sys.stderr)
        return 2
    stages = stages_for_profile(arguments.profile)
    selection = StageSelection.EVERY_STAGE if arguments.all_stages else StageSelection.OUTSTANDING
    print(
        f"ingest: no runtime addressed; processing {len(arguments.uris)} episode(s) "
        f"in this process against {app.data_root}",
        file=sys.stderr,
    )
    try:
        outcomes = run_stages_directly(app, list(arguments.uris), stages, selection=selection)
    except RuntimeError as error:
        # The mass-failure gates, verbatim: the same budgets a scheduled run
        # applies, so a corpus that would fail there fails here too.
        print(f"ingest: {error}", file=sys.stderr)
        _print_ingest_failure_hint()
        return 1
    for outcome in outcomes:
        counts = outcome.counts
        # The skipped count is printed beside the processed one, never folded
        # into it: "0 processed" on a corpus that is entirely up to date should
        # read as nothing left to do, not as nothing having happened.
        already_current = (
            f", {outcome.skipped_as_current} already current" if outcome.skipped_as_current else ""
        )
        print(
            f"{outcome.stage.value}: {counts['processed']} processed, "
            f"{counts['quarantined']} quarantined, {counts['errors']} errors{already_current}"
        )
    if not any(outcome.counts["errors"] for outcome in outcomes):
        return 0
    _print_ingest_failure_hint()
    # Exit 1, per the convention in docs/FORMAT.md: the command ran and found
    # something to report. Under the mass-failure budget an episode that failed
    # is not fatal to the RUN, but it is still a failure, and a `hflow ingest
    # ... && next-step` script has to be able to see it. The budget decides
    # whether to keep going, never whether to report.
    return 1


def _print_ingest_failure_hint() -> None:
    """Where the record of a failed episode lives -- both places it can be.

    Worth spelling out on this path: unlike a scheduled run there is no task
    log behind this executor to go and read, and the two kinds of failure are
    recorded in two different tables. A recording that never canonicalized has
    no catalog row to be, so it lands in the failure ledger; a CHECK that
    crashed leaves an ordinary episode whose step recorded `error`, and
    pointing only at the ledger would send that user looking in an empty table.
    """
    print(
        "ingest: a recording that produced no episode is recorded in "
        "ingest_failures, and a step that crashed on one that did is recorded "
        "on the episode --\n"
        '  hflow curate "SELECT source_uri, failure_kind, message FROM ingest_failures"\n'
        '  hflow curate "SELECT episode_id, check_name, error FROM check_runs '
        "WHERE status = 'error'\"",
        file=sys.stderr,
    )


def _command_ingest(arguments: argparse.Namespace) -> int:
    from posixpath import normpath

    from hflow.runtime import (
        AirflowClientError,
        client_for_bundle,
        client_for_endpoint,
        load_bundle,
    )

    # URIs resolve against the runtime's data root; absolute host paths and
    # ../ escapes cannot work there, so fail before triggering.
    for uri in arguments.uris:
        if uri.startswith("/") or normpath(uri).startswith(".."):
            print(
                f"ingest: {uri!r} is not relative to the data root -- URIs are "
                f"resolved against the workspace this project uses ({_environment_data_root()}), "
                "so name them from there (e.g. `episodes-in/run_0001.mcap`). "
                f"Set $HFLOW_DATA_ROOT or {PROJECT_CONFIG_FILE_NAME}'s data_root "
                "to point at another workspace",
                file=sys.stderr,
            )
            return 2

    try:
        endpoint = _remote_endpoint_for_command(arguments)
    except ValueError as error:
        print(f"ingest: {error}", file=sys.stderr)
        return 2
    if endpoint is not None:
        client = client_for_endpoint(endpoint)
        dag_id = endpoint.dag_id
        watch_location = endpoint.base_url
    else:
        bundle_dir = _found_bundle_dir(arguments.bundle_dir)
        if bundle_dir is None:
            # Nothing addressed: run it here rather than refusing. Starting
            # Airflow is several GB of images and services, far too much to
            # do on someone's behalf inside an ordinary ingest, and far more
            # than a handful of episodes needs.
            return _ingest_in_process(arguments)
        try:
            paths = load_bundle(bundle_dir)
        except (ValueError, FileNotFoundError) as error:
            print(f"ingest: {error}", file=sys.stderr)
            return 2
        client = client_for_bundle(paths)
        dag_id = paths.dag_id
        watch_location = paths.api_base_url
    try:
        dag_run = client.ingest(
            dag_id,
            list(arguments.uris),
            profile=arguments.profile,
            online=arguments.online,
        )
    except AirflowClientError as error:
        print(f"ingest: {error}", file=sys.stderr)
        if error.status == 404:
            if endpoint is None:
                print(
                    "hint: the ingest DAG may still be parsing -- retry in a few "
                    "seconds, or check `docker compose logs airflow-dag-processor`",
                    file=sys.stderr,
                )
            else:
                print(
                    f"hint: no DAG {dag_id!r} at {endpoint.base_url} -- verify --dag-id / "
                    "HFLOW_AIRFLOW_DAG_ID, or retry in a few seconds if the pipeline "
                    "was just deployed",
                    file=sys.stderr,
                )
        return 1
    run_id = dag_run.get("dag_run_id", "<unknown>")
    lane = "online" if arguments.online else "batch"
    print(
        f"triggered {dag_id} run {run_id} over {len(arguments.uris)} episode(s) "
        f"(profile {arguments.profile}, {lane} lane); watch it at {watch_location}"
    )
    return 0


def _command_status(arguments: argparse.Namespace) -> int:
    from hflow.runtime import describe_remote_status, describe_runtime_status, load_bundle

    try:
        endpoint = _remote_endpoint_for_command(arguments)
    except ValueError as error:
        print(f"status: {error}", file=sys.stderr)
        return 2
    if endpoint is not None:
        print(describe_remote_status(endpoint))
        return 0
    try:
        paths = load_bundle(_resolve_bundle_dir(arguments.bundle_dir))
    except (ValueError, FileNotFoundError) as error:
        print(f"status: {error}", file=sys.stderr)
        return 2
    print(describe_runtime_status(paths))
    return 0


def _command_curate(arguments: argparse.Namespace) -> int:
    if (arguments.sql is None) == (arguments.sql_file is None):
        print("curate: pass exactly one of a SQL string or --sql-file", file=sys.stderr)
        return 2
    if arguments.sql_file is not None:
        try:
            sql = arguments.sql_file.read_text()
        except OSError as error:
            # Its own block around the read alone, rather than widening the
            # handler below. `read_text` on a directory raises
            # IsADirectoryError, which subclasses OSError and not
            # FileNotFoundError, so it walked past that handler. Catching
            # OSError here is safe precisely because this block spans one
            # read of one caller-named path: nothing curate does can reach it,
            # so a mid-run filesystem failure is still an unhandled crash.
            print(f"curate: {error}", file=sys.stderr)
            return 2
    else:
        sql = arguments.sql
    try:
        report = curate(
            arguments.catalog,
            sql,
            output=None if arguments.dry_run else arguments.output,
        )
    except (ValueError, FileNotFoundError) as error:
        print(f"curate: {error}", file=sys.stderr)
        return 2
    print(report.summary())
    return 0


def _command_export_snapshot(arguments: argparse.Namespace) -> int:
    from hflow.snapshot import export_dataset_snapshot

    try:
        report = export_dataset_snapshot(
            arguments.catalog,
            arguments.output,
            manifest=arguments.manifest,
            media_mode=arguments.media,
            overwrite=arguments.overwrite,
        )
    except (ValueError, FileNotFoundError, FileExistsError, NotADirectoryError) as error:
        print(f"export snapshot: {error}", file=sys.stderr)
        return 2
    print(report.summary())
    return 0


def _command_doctor(arguments: argparse.Namespace) -> int:
    # Findings, not exceptions, across files as well: an unreadable path is a
    # finding about the corpus, reported in place, so a batch run never loses
    # the reports for the files it could read. Exit precedence (docs/FORMAT.md):
    # 2 only when nothing could be diagnosed; otherwise 1 when any file was
    # non-conforming or unreadable; 0 when everything conformed.
    diagnosed_any = False
    exit_code = 0
    for file in arguments.file:
        try:
            doctor_report = diagnose(file)
        except (ValueError, FileNotFoundError) as error:
            print(f"doctor: {file}\n  [error] unreadable: {error}\n  verdict: NOT CONFORMING")
            exit_code = 1
            continue
        diagnosed_any = True
        print(doctor_report.summary())
        if not doctor_report.conforming:
            exit_code = 1
    return 2 if not diagnosed_any else exit_code


def _command_serve(arguments: argparse.Namespace) -> int:
    try:
        from hflow_server import ServerSettings, ServerStartupError, serve
    except ImportError:
        print(
            "serve: the workspace server ships as a separate package so pipeline "
            "workers never carry it; install it with `uv add hflow-server` "
            "(or `pip install hflow-server`)",
            file=sys.stderr,
        )
        return 2
    try:
        settings = ServerSettings(
            data_root=arguments.data_root,
            host=arguments.host,
            port=arguments.port,
            open_browser=not arguments.no_browser,
            read_only=arguments.read_only,
            # A project that names its pipeline gets the pipeline page without
            # asking; a directory that holds none still serves the catalog,
            # so this stays the one place a missing pipeline is not an error.
            pipeline=arguments.pipeline or _configured_pipeline_spec(),
        )
    except ValueError as error:
        # Its own block, before anything is built: nothing is serving, so this
        # is bad input (2) and not a launch that started and then died (1).
        print(f"serve: {error}", file=sys.stderr)
        return 2
    # A bucket data root has no local directory to check, the same distinction
    # `up` draws for its bundle dir. A local root that EXISTS and is not a
    # directory otherwise serves an empty workspace at a printed URL and says
    # nothing about why it is empty.
    #
    # A root that is not there yet is deliberately allowed through, and `up`
    # agrees (#145). Nothing creates the data root eagerly -- the catalog makes
    # it on first append -- so on a fresh install ./data does not exist until
    # something ingests, and `serve` is a reasonable first command. The server
    # is built for that state: /api/v1/config reports the missing catalog as a
    # capability the frontend hides affordances behind, rather than refusing.
    # An absent root and an empty one look the same to someone who has not
    # ingested yet, so refusing one and serving the other would be a
    # distinction only we can see.
    if not is_bucket_url(settings.data_root):
        local_data_root = Path(settings.data_root)
        if local_data_root.exists() and not local_data_root.is_dir():
            print(
                f"serve: {os.strerror(errno.ENOTDIR)}: {local_data_root}",
                file=sys.stderr,
            )
            return 2
    try:
        serve(settings)
    except ServerStartupError as error:
        # Only the startup failure, not RuntimeError at large: the free-port
        # probe runs before uvicorn binds, so this is still "nothing started".
        # A RuntimeError out of a running server stays an unhandled crash.
        print(f"serve: {error}", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        parser = _build_parser()
    except ValueError as error:
        # Building the parser resolves defaults, which reads hflow.toml. A
        # file that exists and cannot be understood is refused rather than
        # skipped -- falling back to ./data because of a typo would write a
        # corpus into a directory nobody chose -- but it is the user's own
        # just-edited file, so it earns a message and exit 2, not a traceback.
        print(f"hflow: {error}", file=sys.stderr)
        return 2
    arguments = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if arguments.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if arguments.command == "curate":
        return _command_curate(arguments)
    if arguments.command == "dataset":
        if arguments.dataset_command == "create":
            return _command_dataset_create(arguments)
        raise AssertionError(f"unhandled dataset command {arguments.dataset_command!r}")
    if arguments.command == "export":
        if arguments.export_command == "snapshot":
            return _command_export_snapshot(arguments)
        raise AssertionError(f"unhandled export command {arguments.export_command!r}")
    if arguments.command == "stale":
        return _command_stale(arguments)
    if arguments.command == "doctor":
        return _command_doctor(arguments)
    if arguments.command == "manifest":
        return _command_manifest(arguments)
    if arguments.command == "up":
        return _command_up(arguments)
    if arguments.command == "deploy":
        return _command_deploy(arguments)
    if arguments.command == "down":
        return _command_down(arguments)
    if arguments.command == "ingest":
        return _command_ingest(arguments)
    if arguments.command == "status":
        return _command_status(arguments)
    if arguments.command == "serve":
        return _command_serve(arguments)
    raise AssertionError(f"unhandled command {arguments.command!r}")


if __name__ == "__main__":
    sys.exit(main())
