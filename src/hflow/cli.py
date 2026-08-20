"""Command-line entry point.

Subcommands: ``curate``, ``stale``, ``doctor``, the Compose runtime family
``up``/``down``/``ingest``/``status``, and ``deploy`` for bring-your-own
Airflow. Everything the CLI does is a thin call into the library: no behavior
lives only here.
"""

import argparse
import logging
import sys
from pathlib import Path

from hflow import __version__
from hflow.curation import curate, stale_episodes
from hflow.doctor import diagnose
from hflow.runtime._deploy import DEFAULT_DEPLOY_VENV_PYTHON
from hflow.steps import RUN_PROFILES

DEFAULT_DATA_ROOT = Path("./data")
DEFAULT_CATALOG_DIR = "./data/catalog"
DEFAULT_BUNDLE_DIR = DEFAULT_DATA_ROOT / "runtime"
DEFAULT_DEPLOY_OUTPUT_DIR = Path("./deploy")


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
        default=DEFAULT_CATALOG_DIR,
        help=f"catalog directory or object-store prefix (default: {DEFAULT_CATALOG_DIR})",
    )
    curate_parser.add_argument(
        "--output",
        "-o",
        default="./data/manifest.parquet",
        help="manifest path or object-store URL (default: ./data/manifest.parquet)",
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
        default=DEFAULT_CATALOG_DIR,
        help=f"catalog directory or object-store prefix (default: {DEFAULT_CATALOG_DIR})",
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

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="check a file against the canonical-episode convention",
        description=(
            "Validate container integrity, metadata stamps, chunk-group layout, "
            "and in-band video constraints (docs/FORMAT.md, executable form). "
            "Accepts multiple files; exit code 0 when all conform, 1 when any does not."
        ),
    )
    doctor_parser.add_argument(
        "file", nargs="+", help="the local paths or object-store URLs to check"
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
        default=str(DEFAULT_DATA_ROOT),
        help=(
            "host directory mounted at /opt/airflow/data, or a bucket URL "
            "(gs://, s3://, az://) the runtime talks to natively "
            f"(default: {DEFAULT_DATA_ROOT})"
        ),
    )
    up_parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=None,
        help="where to render the bundle (default: <data-root>/runtime; ./runtime for bucket URLs)",
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
        help=f"the rendered bundle to stop (default: {DEFAULT_BUNDLE_DIR}, else ./runtime)",
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
        help=f"the rendered bundle to talk to (default: {DEFAULT_BUNDLE_DIR}, else ./runtime)",
    )
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
        help=f"the rendered bundle to inspect (default: {DEFAULT_BUNDLE_DIR}, else ./runtime)",
    )
    return parser


def _resolve_bundle_dir(bundle_dir_argument: Path | None) -> Path:
    """The bundle a command addresses when ``--bundle-dir`` was not given.

    Local-mode runtimes render at ``<data-root>/runtime`` (default
    ``./data/runtime``); bucket-mode runtimes have no local data root and
    render at ``./runtime``. Try each in that order, falling back to the
    local default so ``load_bundle``'s error names the primary location.
    """
    if bundle_dir_argument is not None:
        return bundle_dir_argument
    for candidate in (DEFAULT_BUNDLE_DIR, Path("runtime")):
        if (candidate / "docker-compose.yaml").is_file():
            return candidate
    return DEFAULT_BUNDLE_DIR


def _parse_pipeline_spec(pipeline_spec: str) -> tuple[Path, str]:
    """Split ``path/to/pipeline.py[:app_variable]`` (default variable: ``app``)."""
    path_part, separator, variable_part = pipeline_spec.rpartition(":")
    if separator and path_part and variable_part.isidentifier():
        return Path(path_part), variable_part
    return Path(pipeline_spec), "app"


def _command_stale(arguments: argparse.Namespace) -> int:
    schema_version: str | None = None
    if arguments.pipeline is not None:
        import importlib.util

        from hflow.app import App
        from hflow.format import EPISODE_FORMAT_VERSION

        pipeline_file, app_variable = _parse_pipeline_spec(arguments.pipeline)
        spec = importlib.util.spec_from_file_location("hflow_user_pipeline", pipeline_file)
        if spec is None or spec.loader is None:
            print(f"stale: cannot import pipeline file {pipeline_file}", file=sys.stderr)
            return 2
        module = importlib.util.module_from_spec(spec)
        try:
            # The pipeline file is arbitrary user code: any exception it
            # raises is a boundary failure of this command, not a crash.
            spec.loader.exec_module(module)
        except Exception as error:
            print(f"stale: importing {pipeline_file} failed: {error}", file=sys.stderr)
            return 2
        app = getattr(module, app_variable, None)
        if not isinstance(app, App):
            print(
                f"stale: {pipeline_file} has no hflow.App named {app_variable!r}",
                file=sys.stderr,
            )
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
    return 0


def _command_up(arguments: argparse.Namespace) -> int:
    from hflow.runtime import (
        RuntimeConfig,
        infer_hflow_source,
        start_runtime,
        started_summary,
    )

    pipeline_file, app_variable = _parse_pipeline_spec(arguments.pipeline)
    hflow_source = (
        arguments.hflow_source if arguments.hflow_source is not None else infer_hflow_source()
    )
    from hflow.storage import is_bucket_url

    config = RuntimeConfig(
        pipeline_file=pipeline_file,
        data_root=arguments.data_root,
        app_variable=app_variable,
        requirements_file=arguments.requirements,
        hflow_source=hflow_source,
    )
    # A bucket data root has no local directory to host the bundle: ./runtime.
    default_bundle_dir = (
        Path("runtime")
        if is_bucket_url(arguments.data_root)
        else Path(arguments.data_root) / "runtime"
    )
    bundle_dir = arguments.bundle_dir if arguments.bundle_dir is not None else default_bundle_dir
    from hflow.runtime import ComposeError

    # Narration goes to stderr so stdout stays exactly the final summary
    # (scripts can capture it; humans see progress on a slow first start).
    def print_progress_to_stderr(message: str) -> None:
        print(f"up: {message}", file=sys.stderr)

    try:
        paths, _ = start_runtime(config, bundle_dir, on_progress=print_progress_to_stderr)
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

    pipeline_file, app_variable = _parse_pipeline_spec(arguments.pipeline)
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

    paths = load_bundle(_resolve_bundle_dir(arguments.bundle_dir))
    compose_down(paths.compose_file, remove_volumes=arguments.volumes)
    print(
        f"runtime at {paths.bundle_dir} stopped{' (volumes removed)' if arguments.volumes else ''}"
    )
    return 0


def _command_ingest(arguments: argparse.Namespace) -> int:
    from posixpath import normpath

    from hflow.runtime import AirflowClientError, client_for_bundle, load_bundle

    # URIs resolve against /opt/airflow/data INSIDE the container; absolute
    # host paths and ../ escapes cannot work there, so fail before triggering.
    for uri in arguments.uris:
        if uri.startswith("/") or normpath(uri).startswith(".."):
            print(
                f"ingest: {uri!r} is not relative to the data root -- URIs are "
                "resolved against the runtime's configured data root "
                "(e.g. `episodes-in/run_0001.mcap`)",
                file=sys.stderr,
            )
            return 2

    paths = load_bundle(_resolve_bundle_dir(arguments.bundle_dir))
    try:
        dag_run = client_for_bundle(paths).ingest(
            paths.dag_id,
            list(arguments.uris),
            profile=arguments.profile,
            online=arguments.online,
        )
    except AirflowClientError as error:
        print(f"ingest: {error}", file=sys.stderr)
        if error.status == 404:
            print(
                "hint: the ingest DAG may still be parsing -- retry in a few "
                "seconds, or check `docker compose logs airflow-dag-processor`",
                file=sys.stderr,
            )
        return 1
    run_id = dag_run.get("dag_run_id", "<unknown>")
    lane = "online" if arguments.online else "batch"
    print(
        f"triggered {paths.dag_id} run {run_id} over {len(arguments.uris)} episode(s) "
        f"(profile {arguments.profile}, {lane} lane); watch it at {paths.api_base_url}"
    )
    return 0


def _command_status(arguments: argparse.Namespace) -> int:
    from hflow.runtime import describe_runtime_status, load_bundle

    paths = load_bundle(_resolve_bundle_dir(arguments.bundle_dir))
    print(describe_runtime_status(paths))
    return 0


def _command_curate(arguments: argparse.Namespace) -> int:
    if (arguments.sql is None) == (arguments.sql_file is None):
        print("curate: pass exactly one of a SQL string or --sql-file", file=sys.stderr)
        return 2
    try:
        sql = arguments.sql if arguments.sql is not None else arguments.sql_file.read_text()
        report = curate(arguments.catalog, sql, output=arguments.output)
    except (ValueError, FileNotFoundError) as error:
        print(f"curate: {error}", file=sys.stderr)
        return 2
    print(report.summary())
    return 0


def _command_doctor(arguments: argparse.Namespace) -> int:
    exit_code = 0
    for file in arguments.file:
        try:
            doctor_report = diagnose(file)
        except (ValueError, FileNotFoundError) as error:
            print(f"doctor: {error}", file=sys.stderr)
            return 2
        print(doctor_report.summary())
        if not doctor_report.conforming:
            exit_code = 1
    return exit_code


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if arguments.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if arguments.command == "curate":
        return _command_curate(arguments)
    if arguments.command == "stale":
        return _command_stale(arguments)
    if arguments.command == "doctor":
        return _command_doctor(arguments)
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
    raise AssertionError(f"unhandled command {arguments.command!r}")


if __name__ == "__main__":
    sys.exit(main())
