"""Command-line entry point.

Subcommands: ``curate``, ``export snapshot``, ``stale``, ``doctor``,
``manifest``, the Compose runtime family ``up``/``down``/``ingest``/``status``,
``deploy`` for bring-your-own Airflow, and ``serve`` for the workspace HTTP
server (a separate ``hflow-server`` package, imported only when invoked).
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
from hflow.app import DATA_ROOT_ENVIRONMENT_VARIABLE, DEFAULT_DATA_ROOT, parse_pipeline_spec
from hflow.curation import curate, stale_episodes
from hflow.doctor import diagnose
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
    """The data root the App itself would resolve: ``HFLOW_DATA_ROOT`` wins.

    CLI defaults derive from the same variable so a shell configured for one
    workspace addresses that workspace's catalog and runtime consistently --
    never a hardcoded ``./data`` beside it.
    """
    return os.environ.get(DATA_ROOT_ENVIRONMENT_VARIABLE) or DEFAULT_DATA_ROOT


def _default_catalog_location() -> str:
    # A string join, not Path: bucket URLs (gs://...) must survive.
    return f"{_environment_data_root().rstrip('/')}/{CATALOG_DIRECTORY_NAME}"


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
            f"(default: $HFLOW_DATA_ROOT/catalog, else {DEFAULT_DATA_ROOT}/catalog)"
        ),
    )
    curate_output_group = curate_parser.add_mutually_exclusive_group()
    curate_output_group.add_argument(
        "--output",
        "-o",
        default=f"{_environment_data_root().rstrip('/')}/manifest.parquet",
        help=(
            "manifest path or object-store URL "
            f"(default: $HFLOW_DATA_ROOT/manifest.parquet, else "
            f"{DEFAULT_DATA_ROOT}/manifest.parquet)"
        ),
    )
    curate_output_group.add_argument(
        "--dry-run",
        action="store_true",
        help="run the query and report row count and coverage without writing a manifest",
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
            f"(default: $HFLOW_DATA_ROOT/catalog, else {DEFAULT_DATA_ROOT}/catalog)"
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
            f"(default: $HFLOW_DATA_ROOT/catalog, else {DEFAULT_DATA_ROOT}/catalog)"
        ),
    )
    stale_group = stale_parser.add_mutually_exclusive_group(required=True)
    stale_group.add_argument(
        "--pipeline",
        help=(
            "pipeline file to compute the current version from, optionally with the "
            "App variable name: path/to/pipeline.py[:app]"
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
        required=True,
        help="pipeline file, optionally with the App variable name: path/to/pipeline.py[:app]",
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
        required=True,
        help="pipeline file, optionally with the App variable name: path/to/pipeline.py[:app]",
    )
    up_parser.add_argument(
        "--data-root",
        type=str,
        default=_environment_data_root(),
        help=(
            "host directory mounted at /opt/airflow/data, or a bucket URL "
            "(gs://, s3://, az://) the runtime talks to natively "
            f"(default: $HFLOW_DATA_ROOT, else {DEFAULT_DATA_ROOT})"
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
        required=True,
        help="pipeline file, optionally with the App variable name: path/to/pipeline.py[:app]",
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
        "--online",
        action="store_true",
        help=(
            "latency-first online lane: process the URIs as one immediate batch "
            "(no bin-packing, no stagger) -- for per-episode runs as data lands"
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


def _resolve_bundle_dir(bundle_dir_argument: Path | None) -> Path:
    """The bundle a command addresses when ``--bundle-dir`` was not given.

    Local-mode runtimes render at ``<data-root>/runtime`` -- resolved from
    ``HFLOW_DATA_ROOT`` when set (matching where ``up`` rendered), else the
    ``./data/runtime`` default; bucket-mode runtimes have no local data root
    and render at ``./runtime``. Try each in that order, falling back to the
    primary candidate so ``load_bundle``'s error names it.
    """
    if bundle_dir_argument is not None:
        return bundle_dir_argument
    environment_data_root = _environment_data_root()
    candidates = [Path(RUNTIME_BUNDLE_DIRECTORY_NAME)]
    if not is_bucket_url(environment_data_root):
        candidates.insert(0, Path(environment_data_root) / RUNTIME_BUNDLE_DIRECTORY_NAME)
    for candidate in candidates:
        if (candidate / "docker-compose.yaml").is_file():
            return candidate
    return candidates[0]


def _import_pipeline_app(pipeline_spec: str) -> "App":
    """Import ``path/to/pipeline.py[:app]`` and return its App, loudly.

    The library owns the contract (:func:`hflow.app.import_pipeline_application`)
    so every vantage that addresses a pipeline by file -- these commands and
    the workspace UI -- resolves it identically.
    """
    from hflow.app import import_pipeline_application

    return import_pipeline_application(pipeline_spec)


def _command_manifest(arguments: argparse.Namespace) -> int:
    try:
        app = _import_pipeline_app(arguments.pipeline)
    except ValueError as error:
        print(f"manifest: {error}", file=sys.stderr)
        return 2
    sys.stdout.write(app.manifest().to_json())
    return 0


def _command_stale(arguments: argparse.Namespace) -> int:
    schema_version: str | None = None
    if arguments.pipeline is not None:
        from hflow.format import EPISODE_FORMAT_VERSION

        try:
            app = _import_pipeline_app(arguments.pipeline)
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

    pipeline_file, app_variable = parse_pipeline_spec(arguments.pipeline)
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

    pipeline_file, app_variable = parse_pipeline_spec(arguments.pipeline)
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
                "resolved against the runtime's configured data root "
                "(e.g. `episodes-in/run_0001.mcap`)",
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
        try:
            paths = load_bundle(_resolve_bundle_dir(arguments.bundle_dir))
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
    try:
        sql = arguments.sql if arguments.sql is not None else arguments.sql_file.read_text()
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
            pipeline=arguments.pipeline,
        )
    except ValueError as error:
        # Its own block, before anything is built: nothing is serving, so this
        # is bad input (2) and not a launch that started and then died (1).
        print(f"serve: {error}", file=sys.stderr)
        return 2
    # A bucket data root has no local directory to check, the same distinction
    # `up` draws for its bundle dir. A local root that is a file, or is not
    # there at all, otherwise serves an empty workspace at a printed URL and
    # says nothing about why it is empty.
    if not is_bucket_url(settings.data_root):
        local_data_root = Path(settings.data_root)
        if not local_data_root.is_dir():
            reason = errno.ENOTDIR if local_data_root.exists() else errno.ENOENT
            print(f"serve: {os.strerror(reason)}: {local_data_root}", file=sys.stderr)
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
    arguments = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if arguments.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if arguments.command == "curate":
        return _command_curate(arguments)
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
