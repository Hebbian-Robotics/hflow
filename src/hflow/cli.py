"""Command-line entry point.

Subcommands: ``curate``, ``catalog ui``, ``dataset create``, ``import lerobot``,
``export snapshot``, ``verify snapshot``, ``verify lerobot-import``, ``stale``,
``doctor``, ``manifest``, ``package``, the Compose runtime family
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

import asyncio
import enum
import errno
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import typer

from hflow import __version__
from hflow.app import (
    DEFAULT_DATA_ROOT,
    default_data_root,
    resolve_pipeline_spec_for_rendering,
)
from hflow.catalog_ui import DEFAULT_CATALOG_UI_PORT
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

# Typer/Click validates an Enum-typed option against its member values before
# any command body runs, exactly like argparse's `choices=`. Built from the
# vocabulary modules own rather than duplicated, so a new profile is
# discoverable here for free.
RunProfile = enum.Enum("RunProfile", {name: name for name in sorted(RUN_PROFILES)}, type=str)
MediaMode = enum.Enum("MediaMode", {"references": "references", "copy": "copy"}, type=str)

_PIPELINE_OPTION_HELP = (
    "pipeline file, optionally with the App variable name: "
    "path/to/pipeline.py[:app]. Defaults to hflow.toml's `pipeline`, "
    f"else ./{DEFAULT_PIPELINE_FILE_NAME}"
)


def _catalog_option_help() -> str:
    return (
        "catalog directory or object-store prefix "
        f"(default: $HFLOW_DATA_ROOT, else {PROJECT_CONFIG_FILE_NAME}'s data_root, "
        f"else {DEFAULT_DATA_ROOT} -- plus /catalog)"
    )


def _bundle_dir_help() -> str:
    return (
        "the rendered bundle to talk to (default: $HFLOW_DATA_ROOT/runtime, "
        f"else {DEFAULT_DATA_ROOT}/runtime; ./runtime for bucket data roots)"
    )


def _parameter_was_explicit(ctx: typer.Context, parameter_name: str) -> bool:
    """Whether ``parameter_name`` was actually typed on argv, not defaulted.

    Reproduces argparse's mutually-exclusive-group semantics for an option
    that (unlike a plain ``store_true`` flag) always carries a non-``None``
    default: the conflict only exists when both were explicitly given.
    """
    source = ctx.get_parameter_source(parameter_name)
    return source is not None and source.name == "COMMANDLINE"


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


def _remote_endpoint_for_command(
    airflow_url: str | None, bundle_dir: Path | None, dag_id: str | None
) -> "RemoteRuntimeEndpoint | None":
    """The remote endpoint this command addresses, or ``None`` for local.

    An explicit ``--bundle-dir`` keeps the command local even when
    ``HFLOW_AIRFLOW_URL`` is exported; an explicit ``--airflow-url`` wins the
    other way. Raises ``ValueError`` when a remote resolution is incomplete.
    """
    from hflow.runtime import resolve_remote_endpoint

    if airflow_url is None and bundle_dir is not None:
        return None
    return resolve_remote_endpoint(airflow_url=airflow_url, dag_id=dag_id)


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


def _command_dataset_create(
    name: str, pipeline: str | None, sql: str | None, print_sql: bool
) -> int:
    from hflow.dataset import create_dataset, default_dataset_sql

    try:
        app = _import_pipeline_app(_require_pipeline_spec(pipeline))
    except ValueError as error:
        print(f"dataset create: {error}", file=sys.stderr)
        return 2
    if print_sql:
        # The policy is never hidden: it can always be read, edited, and
        # handed back through --sql or `hflow curate`.
        print(sql if sql is not None else default_dataset_sql(app))
        return 0
    try:
        dataset = create_dataset(app, name, sql=sql)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        print(f"dataset create: {error}", file=sys.stderr)
        return 2
    print(dataset.summary())
    return 0


def _command_import_lerobot(
    repo: str,
    revision: str,
    output_dir: str,
    camera_keys: list[str] | None,
    episode_index: int | None,
) -> int:
    from hflow.importers.lerobot import DEFAULT_CAMERA_KEY, import_lerobot_dataset

    resolved_camera_keys = camera_keys or [DEFAULT_CAMERA_KEY]
    bucket_destination = is_bucket_url(output_dir)
    bucket_storage_error: type[BaseException] | None = None
    if bucket_destination:
        # Import while the optional extra is already known to be required for
        # this path, so a later handler cannot mask an unrelated failure with
        # ModuleNotFoundError from this import.
        try:
            from obstore.exceptions import BaseError as BucketStorageError
        except ModuleNotFoundError:
            bucket_storage_error = None
        else:
            bucket_storage_error = BucketStorageError

    try:
        output_uris = import_lerobot_dataset(
            dataset_repo=repo,
            revision=revision,
            output_dir=output_dir,
            episode_index=episode_index,
            camera_keys=resolved_camera_keys,
        )
    except (ValueError, RuntimeError, OSError, ModuleNotFoundError) as error:
        print(f"import lerobot: {error}", file=sys.stderr)
        return 2
    except Exception as error:
        if bucket_storage_error is not None and isinstance(error, bucket_storage_error):
            print(f"import lerobot: {error}", file=sys.stderr)
            return 2
        raise
    print(f"import lerobot: converted {len(output_uris)} episode(s)")
    return 0


def _command_manifest(pipeline: str | None) -> int:
    try:
        app = _import_pipeline_app(_require_pipeline_spec(pipeline))
    except ValueError as error:
        print(f"manifest: {error}", file=sys.stderr)
        return 2
    sys.stdout.write(app.manifest().to_json())
    return 0


def _command_package_build(
    package_root: Path,
    package_name: str | None,
    module_names: list[str] | None,
    jobs: int,
    output_dir: Path,
) -> int:
    from hflow.packaging import (
        CythonOverlayApplyError,
        CythonOverlayBuildConfig,
        CythonOverlayBuildError,
        CythonOverlayManifestError,
        build_cython_overlay,
    )

    try:
        manifest = build_cython_overlay(
            CythonOverlayBuildConfig(
                package_root=package_root,
                package_name=package_name,
                module_names=tuple(module_names) if module_names is not None else None,
                jobs=jobs,
            ),
            output_dir,
        )
    except (
        CythonOverlayApplyError,
        CythonOverlayBuildError,
        CythonOverlayManifestError,
        FileExistsError,
        OSError,
    ) as error:
        print(f"package build: {error}", file=sys.stderr)
        return 2
    print(
        f"native overlay: {output_dir} ({len(manifest.artifacts)} modules, {manifest.bundle_digest})"
    )
    return 0


def _command_package_verify(overlay_dir: Path, target_package_root: Path | None) -> int:
    from hflow.packaging import (
        CythonOverlayApplyError,
        CythonOverlayBuildError,
        CythonOverlayManifestError,
        verify_cython_overlay,
    )

    try:
        outcome = verify_cython_overlay(overlay_dir, target_package_root=target_package_root)
    except (
        CythonOverlayApplyError,
        CythonOverlayBuildError,
        CythonOverlayManifestError,
        FileExistsError,
        OSError,
    ) as error:
        print(f"package verify: {error}", file=sys.stderr)
        return 2
    if outcome.succeeded:
        checked_target = (
            f" and applied package {target_package_root}" if target_package_root is not None else ""
        )
        print(f"native overlay verified: {overlay_dir}{checked_target}")
        return 0
    for issue in outcome.issues:
        issue_context = f": {issue.path}" if issue.path is not None else ""
        print(f"package verify: {issue.code.value}{issue_context}", file=sys.stderr)
    return 1


def _command_package_apply(overlay_dir: Path, target_package_root: Path) -> int:
    from hflow.packaging import (
        CythonOverlayApplyError,
        CythonOverlayBuildError,
        CythonOverlayManifestError,
        apply_cython_overlay,
    )

    try:
        manifest = apply_cython_overlay(overlay_dir, target_package_root)
    except (
        CythonOverlayApplyError,
        CythonOverlayBuildError,
        CythonOverlayManifestError,
        FileExistsError,
        OSError,
    ) as error:
        print(f"package apply: {error}", file=sys.stderr)
        return 2
    print(
        f"native overlay applied: {target_package_root} "
        f"({len(manifest.artifacts)} modules, {manifest.bundle_digest})"
    )
    return 0


def _command_stale(
    catalog: str, pipeline: str | None, pipeline_version: str | None, exit_code_flag: bool
) -> int:
    schema_version: str | None = None
    if pipeline_version is None:
        from hflow.format import EPISODE_FORMAT_VERSION

        try:
            app = _import_pipeline_app(_require_pipeline_spec(pipeline))
        except ValueError as error:
            print(f"stale: {error}", file=sys.stderr)
            return 2
        resolved_pipeline_version = app.pipeline_version
        # A pipeline defines the whole current target, format version included.
        schema_version = EPISODE_FORMAT_VERSION
    else:
        resolved_pipeline_version = pipeline_version

    try:
        stale = stale_episodes(
            catalog,
            pipeline_version=resolved_pipeline_version,
            schema_version=schema_version,
        )
    except (ValueError, FileNotFoundError) as error:
        print(f"stale: {error}", file=sys.stderr)
        return 2
    for episode in stale:
        print(episode.source_uri if episode.source_uri is not None else episode.uri)
    print(
        f"stale: {len(stale)} episode(s) behind pipeline_version {resolved_pipeline_version}"
        + (f" / schema_version {schema_version}" if schema_version is not None else ""),
        file=sys.stderr,
    )
    if exit_code_flag and stale:
        return 1
    return 0


def _command_up(
    pipeline: str | None,
    data_root: str,
    bundle_dir: Path | None,
    api_port: int,
    hflow_source_argument: Path | None,
    requirements: Path | None,
    pass_env: list[str],
) -> int:
    from hflow.runtime import (
        RuntimeConfig,
        infer_hflow_source,
        start_runtime,
        started_summary,
    )

    try:
        pipeline_file, app_variable = resolve_pipeline_spec_for_rendering(
            _require_pipeline_spec(pipeline)
        )
    except ValueError as error:
        print(f"up: {error}", file=sys.stderr)
        return 2
    hflow_source = (
        hflow_source_argument if hflow_source_argument is not None else infer_hflow_source()
    )
    try:
        config = RuntimeConfig(
            pipeline_file=pipeline_file,
            data_root=data_root,
            app_variable=app_variable,
            requirements_file=requirements,
            hflow_source=hflow_source,
            api_port=api_port,
            passthrough_environment_variables=tuple(pass_env),
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
    if not is_bucket_url(data_root):
        local_data_root = Path(data_root)
        if local_data_root.exists() and not local_data_root.is_dir():
            print(
                f"up: {os.strerror(errno.ENOTDIR)}: {local_data_root}",
                file=sys.stderr,
            )
            return 2
    # A bucket data root has no local directory to host the bundle: ./runtime.
    default_bundle_dir = (
        Path(RUNTIME_BUNDLE_DIRECTORY_NAME)
        if is_bucket_url(data_root)
        else Path(data_root) / RUNTIME_BUNDLE_DIRECTORY_NAME
    )
    resolved_bundle_dir = bundle_dir if bundle_dir is not None else default_bundle_dir
    from hflow.runtime import ComposeError

    # Narration goes to stderr so stdout stays exactly the final summary
    # (scripts can capture it; humans see progress on a slow first start).
    def print_progress_to_stderr(message: str) -> None:
        print(f"up: {message}", file=sys.stderr)

    try:
        paths, _ = start_runtime(config, resolved_bundle_dir, on_progress=print_progress_to_stderr)
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
                    f"containers may still be running for the bundle at {resolved_bundle_dir}.",
                    f"  inspect:   hflow status --bundle-dir {resolved_bundle_dir}",
                    f"  logs:      docker compose --file {resolved_bundle_dir}/docker-compose.yaml logs <service>",
                    f"  tear down: hflow down --bundle-dir {resolved_bundle_dir}",
                ]
            ),
            file=sys.stderr,
        )
        return 1
    print(started_summary(paths))
    return 0


def _command_deploy(
    pipeline: str | None,
    data_root_uri: str,
    output_dir: Path,
    requirements: Path | None,
    venv_python: str,
    pass_env: list[str],
) -> int:
    from hflow.runtime._deploy import DeployConfig, render_deploy_bundle

    try:
        pipeline_file, app_variable = resolve_pipeline_spec_for_rendering(
            _require_pipeline_spec(pipeline)
        )
    except ValueError as error:
        print(f"deploy: {error}", file=sys.stderr)
        return 2
    try:
        config = DeployConfig(
            pipeline_file=pipeline_file,
            data_root_uri=data_root_uri,
            app_variable=app_variable,
            requirements_file=requirements,
            venv_python_path=venv_python,
            passthrough_environment_variables=tuple(pass_env),
        )
        paths = render_deploy_bundle(config, output_dir)
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


def _command_down(bundle_dir: Path | None, volumes: bool) -> int:
    from hflow.runtime import compose_down, load_bundle

    try:
        paths = load_bundle(_resolve_bundle_dir(bundle_dir))
    except (ValueError, FileNotFoundError) as error:
        print(f"down: {error}", file=sys.stderr)
        return 2
    compose_down(paths.compose_file, remove_volumes=volumes)
    print(f"runtime at {paths.bundle_dir} stopped{' (volumes removed)' if volumes else ''}")
    return 0


def _ingest_in_process(
    pipeline: str | None,
    uris: list[str],
    profile: str,
    all_stages: bool,
    step_names: list[str] | None,
) -> int:
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
        app = _import_pipeline_app(_require_pipeline_spec(pipeline))
    except ValueError as error:
        print(f"ingest: {error}", file=sys.stderr)
        return 2
    stages = stages_for_profile(profile)
    selection = StageSelection.EVERY_STAGE if all_stages else StageSelection.OUTSTANDING
    print(
        f"ingest: no runtime addressed; processing {len(uris)} episode(s) "
        f"in this process against {app.data_root}",
        file=sys.stderr,
    )
    try:
        outcomes = asyncio.run(
            run_stages_directly(
                app,
                list(uris),
                stages,
                selection=selection,
                step_names=step_names,
            )
        )
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


def _command_ingest(
    uris: list[str],
    bundle_dir: Path | None,
    airflow_url: str | None,
    dag_id: str | None,
    profile: str,
    all_stages: bool,
    step_names: list[str] | None,
    online: bool,
    pipeline: str | None,
) -> int:
    from hflow.runtime import (
        AirflowClientError,
        client_for_bundle,
        client_for_endpoint,
        load_bundle,
        parse_data_root_relative_uri,
    )

    # URIs resolve against the runtime's data root; absolute host paths and
    # ../ escapes cannot work there, so fail before triggering.
    try:
        parsed_uris = [parse_data_root_relative_uri(uri) for uri in uris]
    except ValueError as error:
        print(
            f"ingest: {error} -- URIs are resolved against the workspace this project uses "
            f"({_environment_data_root()}), so name them from there "
            "(e.g. `episodes-in/run_0001.mcap`). "
            f"Set $HFLOW_DATA_ROOT or {PROJECT_CONFIG_FILE_NAME}'s data_root",
            file=sys.stderr,
        )
        return 2

    try:
        endpoint = _remote_endpoint_for_command(airflow_url, bundle_dir, dag_id)
    except ValueError as error:
        print(f"ingest: {error}", file=sys.stderr)
        return 2
    if endpoint is not None:
        client = client_for_endpoint(endpoint)
        resolved_dag_id = endpoint.dag_id
        watch_location = endpoint.base_url
    else:
        found_bundle_dir = _found_bundle_dir(bundle_dir)
        if found_bundle_dir is None:
            # Nothing addressed: run it here rather than refusing. Starting
            # Airflow is several GB of images and services, far too much to
            # do on someone's behalf inside an ordinary ingest, and far more
            # than a handful of episodes needs.
            return _ingest_in_process(
                pipeline, [str(uri) for uri in parsed_uris], profile, all_stages, step_names
            )
        try:
            paths = load_bundle(found_bundle_dir)
        except (ValueError, FileNotFoundError) as error:
            print(f"ingest: {error}", file=sys.stderr)
            return 2
        client = client_for_bundle(paths)
        resolved_dag_id = paths.dag_id
        watch_location = paths.api_base_url
    with client:
        try:
            if step_names is None:
                dag_run = client.ingest(
                    resolved_dag_id,
                    [str(uri) for uri in parsed_uris],
                    profile=profile,
                    online=online,
                )
            else:
                dag_run = client.ingest(
                    resolved_dag_id,
                    [str(uri) for uri in parsed_uris],
                    profile=profile,
                    online=online,
                    step_names=step_names,
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
                        f"hint: no DAG {resolved_dag_id!r} at {endpoint.base_url} -- verify --dag-id / "
                        "HFLOW_AIRFLOW_DAG_ID, or retry in a few seconds if the pipeline "
                        "was just deployed",
                        file=sys.stderr,
                    )
            return 1
    run_id = dag_run.dag_run_id or "<unknown>"
    lane = "online" if online else "batch"
    print(
        f"triggered {resolved_dag_id} run {run_id} over {len(parsed_uris)} episode(s) "
        f"(profile {profile}, {lane} lane); watch it at {watch_location}"
    )
    return 0


def _command_status(bundle_dir: Path | None, airflow_url: str | None, dag_id: str | None) -> int:
    from hflow.runtime import describe_remote_status, describe_runtime_status, load_bundle

    try:
        endpoint = _remote_endpoint_for_command(airflow_url, bundle_dir, dag_id)
    except ValueError as error:
        print(f"status: {error}", file=sys.stderr)
        return 2
    if endpoint is not None:
        print(describe_remote_status(endpoint))
        return 0
    try:
        paths = load_bundle(_resolve_bundle_dir(bundle_dir))
    except (ValueError, FileNotFoundError) as error:
        print(f"status: {error}", file=sys.stderr)
        return 2
    print(describe_runtime_status(paths))
    return 0


def _command_curate(
    sql: str | None, sql_file: Path | None, catalog: str, output: str, dry_run: bool
) -> int:
    if (sql is None) == (sql_file is None):
        print("curate: pass exactly one of a SQL string or --sql-file", file=sys.stderr)
        return 2
    if sql_file is not None:
        try:
            resolved_sql = sql_file.read_text()
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
        # The XOR check above guarantees sql is not None here.
        assert sql is not None
        resolved_sql = sql
    try:
        report = curate(catalog, resolved_sql, output=None if dry_run else output)
    except (ValueError, FileNotFoundError) as error:
        print(f"curate: {error}", file=sys.stderr)
        return 2
    print(report.summary())
    return 0


def _command_export_snapshot(
    catalog: str, manifest: str | None, output: Path, media: str, overwrite: bool
) -> int:
    from hflow.snapshot import export_dataset_snapshot

    try:
        report = export_dataset_snapshot(
            catalog,
            output,
            manifest=manifest,
            media_mode=media,
            overwrite=overwrite,
        )
    except (ValueError, FileNotFoundError, FileExistsError, NotADirectoryError) as error:
        print(f"export snapshot: {error}", file=sys.stderr)
        return 2
    print(report.summary())
    return 0


def _command_verify_snapshot(directory: str) -> int:
    from hflow.snapshot import verify_dataset_snapshot
    from hflow.verification import exit_code_for

    try:
        report = verify_dataset_snapshot(Path(directory))
    except (ValueError, FileNotFoundError, NotADirectoryError, OSError) as error:
        print(f"verify snapshot: {error}", file=sys.stderr)
        return 2
    if report.ok:
        print("verified: every receipted file matches its size and sha256")
        return exit_code_for(report)
    print(f"not verified: {len(report.findings)} finding(s)")
    for finding in report.findings:
        print(f"  [{finding.reason}] {finding.uri}")
        print(f"    {finding.detail}")
    return exit_code_for(report)


def _command_verify_lerobot_import(data_root: str) -> int:
    from hflow.importers.lerobot_verify import verify_lerobot_import
    from hflow.verification import exit_code_for

    try:
        report = verify_lerobot_import(data_root)
    except (ValueError, OSError, ModuleNotFoundError) as error:
        print(f"verify lerobot-import: {error}", file=sys.stderr)
        return 2

    if report.ok:
        print("verify lerobot-import: ok")
        return 0

    for finding in report.findings:
        print(
            f"verify lerobot-import: {finding.reason}: {finding.uri}: {finding.detail}",
            file=sys.stderr,
        )
    if not report.findings:
        print("verify lerobot-import: unverifiable: no prepared-manifest receipt", file=sys.stderr)
    return exit_code_for(report)


def _command_doctor(files: list[str]) -> int:
    # Findings, not exceptions, across files as well: an unreadable path is a
    # finding about the corpus, reported in place, so a batch run never loses
    # the reports for the files it could read. Exit precedence (docs/FORMAT.md):
    # 2 only when nothing could be diagnosed; otherwise 1 when any file was
    # non-conforming or unreadable; 0 when everything conformed.
    diagnosed_any = False
    exit_code = 0
    for file in files:
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


def _command_catalog_ui(catalog: str, port: int, no_browser: bool) -> int:
    from hflow.catalog_ui import (
        CatalogUiSettings,
        CatalogUiStartupError,
        serve_catalog_ui,
    )

    if not is_bucket_url(catalog):
        local_catalog_root = Path(catalog)
        if local_catalog_root.exists() and not local_catalog_root.is_dir():
            print(
                f"catalog ui: {os.strerror(errno.ENOTDIR)}: {local_catalog_root}",
                file=sys.stderr,
            )
            return 2
    bucket_catalog = is_bucket_url(catalog)
    bucket_storage_error: type[BaseException] | None = None
    if bucket_catalog:
        # Import while the optional extra is already known to be required for
        # this path, so a later handler cannot mask an unrelated failure with
        # ModuleNotFoundError from this import.
        try:
            from obstore.exceptions import BaseError as BucketStorageError
        except ModuleNotFoundError:
            bucket_storage_error = None
        else:
            bucket_storage_error = BucketStorageError

    try:
        settings = CatalogUiSettings(
            catalog_root=catalog,
            port=port,
            open_browser=not no_browser,
        )
        serve_catalog_ui(settings)
    except (
        CatalogUiStartupError,
        OSError,
        ValueError,
        ModuleNotFoundError,
    ) as error:
        print(f"catalog ui: {error}", file=sys.stderr)
        return 2
    except Exception as error:
        if bucket_storage_error is not None and isinstance(error, bucket_storage_error):
            print(f"catalog ui: {error}", file=sys.stderr)
            return 2
        raise
    return 0


def _command_serve(
    data_root: str, host: str, port: int, no_browser: bool, read_only: bool, pipeline: str | None
) -> int:
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
            data_root=data_root,
            host=host,
            port=port,
            open_browser=not no_browser,
            read_only=read_only,
            # A project that names its pipeline gets the pipeline page without
            # asking; a directory that holds none still serves the catalog,
            # so this stays the one place a missing pipeline is not an error.
            pipeline=pipeline or _configured_pipeline_spec(),
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


def _register_curate_command(app: typer.Typer, result_holder: list[int]) -> None:
    @app.command(
        "curate",
        short_help="run a SQL query over the episode catalog and write manifest.parquet",
        help=(
            "Run any SELECT over the catalog views (the wide 'episodes' view "
            "covers everyday cuts) and write the result as a Parquet manifest."
        ),
    )
    def curate_command(
        ctx: typer.Context,
        sql: str | None = typer.Argument(None, help="the SELECT to run (or pass --sql-file)"),
        sql_file: Path | None = typer.Option(
            None, "--sql-file", help="read the SELECT from a file instead of the command line"
        ),
        catalog: str = typer.Option(
            _default_catalog_location(), "--catalog", help=_catalog_option_help()
        ),
        output: str = typer.Option(
            f"{_environment_data_root().rstrip('/')}/manifest.parquet",
            "--output",
            "-o",
            help=(
                "manifest path or object-store URL "
                f"(default: $HFLOW_DATA_ROOT, else {PROJECT_CONFIG_FILE_NAME}'s data_root, "
                f"else {DEFAULT_DATA_ROOT} -- plus /manifest.parquet)"
            ),
        ),
        dry_run: bool = typer.Option(
            False,
            "--dry-run",
            help="run the query and report row count and coverage without writing a manifest",
        ),
    ) -> int:
        if dry_run and _parameter_was_explicit(ctx, "output"):
            print("curate: --output/-o and --dry-run are mutually exclusive", file=sys.stderr)
            code = 2
        else:
            code = _command_curate(sql, sql_file, catalog, output, dry_run)
        result_holder.append(code)
        return code


def _register_catalog_group(app: typer.Typer, result_holder: list[int]) -> None:
    catalog_app = typer.Typer(no_args_is_help=False)

    @catalog_app.callback()
    def _catalog_root() -> None:
        # A no-op callback: without one, Typer collapses a Typer() instance
        # holding exactly one command into a bare command, dropping the
        # `catalog ui` subcommand name entirely.
        pass

    @catalog_app.command(
        "ui",
        help=(
            "Start DuckDB's browser UI over an HFlow catalog. The UI starts "
            "even when the catalog is empty, then refreshes its views after the "
            "first completed append. Local catalogs may be created on startup; "
            "bucket catalogs must already exist and are read-only."
        ),
    )
    def catalog_ui_command(
        catalog: str = typer.Option(
            _default_catalog_location(),
            "--catalog",
            help=(
                "catalog root: local directory or object-store prefix "
                f"(s3://, gs://, az://; default: $HFLOW_DATA_ROOT, else "
                f"{PROJECT_CONFIG_FILE_NAME}'s data_root, else {DEFAULT_DATA_ROOT} -- plus /catalog)"
            ),
        ),
        port: int = typer.Option(
            DEFAULT_CATALOG_UI_PORT,
            "--port",
            help=f"local DuckDB UI port (default {DEFAULT_CATALOG_UI_PORT})",
        ),
        no_browser: bool = typer.Option(
            False,
            "--no-browser",
            help="do not open a browser after starting (headless or remote use)",
        ),
    ) -> int:
        code = _command_catalog_ui(catalog, port, no_browser)
        result_holder.append(code)
        return code

    app.add_typer(
        catalog_app,
        name="catalog",
        short_help="inspect and explore the Parquet catalog",
        help=(
            "Group commands for inspecting and exploring the append-only Parquet "
            "catalog. Use `ui` to open DuckDB's local browser interface over the "
            "HFlow views."
        ),
    )


def _register_dataset_group(app: typer.Typer, result_holder: list[int]) -> None:
    dataset_app = typer.Typer(no_args_is_help=False)

    @dataset_app.callback()
    def _dataset_root() -> None:
        pass

    @dataset_app.command(
        "create",
        help=(
            "Select the current generation of every source recording that is not "
            "quarantined, was produced by this pipeline's current transform, and "
            "has every registered step recorded at its current version. Writes "
            "manifests/<name>-<timestamp>.parquet plus a .json recording the "
            "effective SQL and the versions it required. Importing EXECUTES the "
            "pipeline file, so run this in the pipeline's own environment."
        ),
    )
    def dataset_create_command(
        name: str = typer.Argument(
            ..., help="a name for the dataset; slugified into the manifest's filename"
        ),
        pipeline: str | None = typer.Option(None, "--pipeline", help=_PIPELINE_OPTION_HELP),
        sql: str | None = typer.Option(
            None,
            "--sql",
            help=(
                "replace the default policy with your own SELECT, keeping the "
                "immutable artifact and the provenance record"
            ),
        ),
        print_sql: bool = typer.Option(
            False, "--print-sql", help="print the SQL this would run and exit, writing nothing"
        ),
    ) -> int:
        code = _command_dataset_create(name, pipeline, sql, print_sql)
        result_holder.append(code)
        return code

    app.add_typer(
        dataset_app,
        name="dataset",
        short_help="create version-pinned dataset manifests from the pipeline's own policy",
        help=(
            "Group commands that turn the pipeline's policy into version-pinned "
            "dataset manifests. Use `create` to select episodes and write a "
            "Parquet manifest plus its provenance sidecar."
        ),
    )


def _register_import_group(app: typer.Typer, result_holder: list[int]) -> None:
    import_app = typer.Typer(no_args_is_help=False)

    @import_app.callback()
    def _import_root() -> None:
        pass

    @import_app.command(
        "lerobot",
        help=(
            "Import LeRobot Dataset v3 video cameras and fixed-width float32 "
            "state/action vectors as canonical MCAP episodes. The source revision "
            "is resolved to an immutable commit and recorded as provenance."
        ),
    )
    def import_lerobot_command(
        repo: str = typer.Option(
            ..., "--repo", help="Hugging Face dataset repository, for example lerobot/pusht"
        ),
        revision: str = typer.Option(
            "main", "--revision", help="branch, tag, or commit to resolve (default: main)"
        ),
        output_dir: str = typer.Option(
            ...,
            "--output-dir",
            help=(
                "destination data root: local directory or object-store prefix "
                "(s3://, gs://, az://); episodes publish under landing/"
            ),
        ),
        camera_keys: list[str] | None = typer.Option(
            None,
            "--camera",
            "--camera-key",
            help=(
                "video feature to import; repeat for multiple cameras (default: "
                "observation.image; comma-separated values are also accepted)"
            ),
        ),
        episode_index: int | None = typer.Option(
            None,
            "--episode-index",
            help="zero-based episode index to import (default: every episode)",
        ),
    ) -> int:
        code = _command_import_lerobot(repo, revision, output_dir, camera_keys, episode_index)
        result_holder.append(code)
        return code

    app.add_typer(
        import_app,
        name="import",
        short_help="import supported robotics datasets as canonical MCAP episodes",
        help=(
            "Group commands that import supported source datasets into HFlow's "
            "canonical MCAP episode boundary."
        ),
    )


def _register_export_group(app: typer.Typer, result_holder: list[int]) -> None:
    export_app = typer.Typer(no_args_is_help=False)

    @export_app.callback()
    def _export_root() -> None:
        pass

    @export_app.command(
        "snapshot",
        help=(
            "Snapshot selected episodes, measurements, artifact media, check runs, "
            "tags, and intervals into a local directory of standard Parquet files."
        ),
    )
    def export_snapshot_command(
        catalog: str = typer.Option(
            _default_catalog_location(), "--catalog", help=_catalog_option_help()
        ),
        manifest: str | None = typer.Option(
            None,
            "--manifest",
            help=(
                "optional local Parquet file or object-store URL containing episode_id; "
                "without it, export every latest catalog episode"
            ),
        ),
        output: Path = typer.Option(..., "--output", "-o", help="local directory to create"),
        media: MediaMode = typer.Option(
            MediaMode.references,
            "--media",
            help=(
                "preserve artifact URIs, or copy artifacts under the export's assets/ "
                "directory (default: references)"
            ),
        ),
        overwrite: bool = typer.Option(
            False, "--overwrite", help="atomically replace an existing export directory"
        ),
    ) -> int:
        code = _command_export_snapshot(catalog, manifest, output, media.value, overwrite)
        result_holder.append(code)
        return code

    app.add_typer(
        export_app,
        name="export",
        short_help="export catalog selections in portable downstream formats",
        help=(
            "Group commands for exporting catalog selections in portable downstream "
            "formats. Use `snapshot` to write a tool-neutral directory of Parquet "
            "tables and optional media assets."
        ),
    )


def _register_verify_group(app: typer.Typer, result_holder: list[int]) -> None:
    verify_app = typer.Typer(no_args_is_help=False)

    @verify_app.callback()
    def _verify_root() -> None:
        pass

    @verify_app.command(
        "snapshot",
        help=(
            "Re-reads every table and copied asset named in the snapshot's "
            "integrity receipt and reports bytes changed, files missing, and "
            "size mismatches. Exit 0 clean, 1 damaged, 2 unreadable input, "
            "3 no receipt (unverifiable)."
        ),
    )
    def verify_snapshot_command(
        directory: str = typer.Argument(
            ..., help="the delivered dataset snapshot directory to verify"
        ),
    ) -> int:
        code = _command_verify_snapshot(directory)
        result_holder.append(code)
        return code

    @verify_app.command(
        "lerobot-import",
        help=(
            "Read schema-3 prepared-manifest.json under a data root and check every "
            "episode receipt against landing/<basename> under that root. Unlisted "
            "files under landing/ are ignored. Exit 0 clean, 1 damaged, 2 unreadable "
            "input, 3 unverifiable (no prepared-manifest)."
        ),
    )
    def verify_lerobot_import_command(
        data_root: str = typer.Argument(
            ...,
            help=(
                "data root that holds prepared-manifest.json and landing/: local "
                "directory or object-store prefix (s3://, gs://, az://)"
            ),
        ),
    ) -> int:
        code = _command_verify_lerobot_import(data_root)
        result_holder.append(code)
        return code

    app.add_typer(
        verify_app,
        name="verify",
        short_help="verify a delivery against its receipt",
        help=(
            "Group commands for verifying deliveries against their recorded "
            "receipts. Use `snapshot` to re-check a delivered dataset snapshot "
            "directory against the integrity receipt inside its format.json. "
            "Use `lerobot-import` to re-check a LeRobot prepared-manifest delivery."
        ),
    )


def _register_stale_command(app: typer.Typer, result_holder: list[int]) -> None:
    @app.command(
        "stale",
        short_help="list episodes whose latest cataloged run predates the current pipeline version",
        help=(
            "Print the source URI of every episode whose latest cataloged run was "
            "produced by a different pipeline (and format) version -- one per line "
            "on stdout, ready to pipe back into `hflow ingest` for selective "
            "reprocessing. The summary goes to stderr."
        ),
    )
    def stale_command(
        catalog: str = typer.Option(
            _default_catalog_location(), "--catalog", help=_catalog_option_help()
        ),
        # Not required: without either flag the pipeline is resolved from
        # hflow.toml or ./pipeline.py, exactly as the other commands do.
        pipeline: str | None = typer.Option(
            None,
            "--pipeline",
            help=(
                "pipeline file to compute the current version from, optionally with the "
                "App variable name: path/to/pipeline.py[:app]. Defaults to hflow.toml's "
                f"`pipeline`, else ./{DEFAULT_PIPELINE_FILE_NAME}"
            ),
        ),
        pipeline_version: str | None = typer.Option(
            None,
            "--pipeline-version",
            help="compare against this pipeline_version hash directly (no pipeline import)",
        ),
        exit_code_flag: bool = typer.Option(
            False,
            "--exit-code",
            help=(
                "exit 1 when at least one stale episode is found (like "
                "`git diff --exit-code`), so CI can gate on it; without this flag the "
                "command always exits 0"
            ),
        ),
    ) -> int:
        if pipeline is not None and pipeline_version is not None:
            print("stale: pass at most one of --pipeline or --pipeline-version", file=sys.stderr)
            code = 2
        else:
            code = _command_stale(catalog, pipeline, pipeline_version, exit_code_flag)
        result_holder.append(code)
        return code


def _register_doctor_command(app: typer.Typer, result_holder: list[int]) -> None:
    @app.command(
        "doctor",
        short_help="check a file against the canonical-episode convention",
        help=(
            "Validate container integrity, metadata stamps, chunk-group layout, "
            "and in-band video constraints (docs/FORMAT.md, executable form). "
            "Accepts multiple files, each reported in order whatever happens "
            "to the others. Exit 0 when all conform, 1 when any file is "
            "non-conforming or could not be read, 2 when no file could be "
            "diagnosed (nothing useful happened)."
        ),
    )
    def doctor_command(
        files: list[str] = typer.Argument(
            ..., help="the local paths or object-store URLs to check"
        ),
    ) -> int:
        code = _command_doctor(files)
        result_holder.append(code)
        return code


def _register_manifest_command(app: typer.Typer, result_holder: list[int]) -> None:
    @app.command(
        "manifest",
        short_help="print the pipeline's manifest (steps, versions, requirements) as JSON",
        help=(
            "Import the pipeline file and print its manifest -- step names, "
            "explicit versions, gate flags, resource requirements, and version "
            "stamps -- as JSON on stdout. This is the metadata a pipeline "
            "crosses a control boundary as. Importing EXECUTES the pipeline "
            "file, so run this in the pipeline's own environment."
        ),
    )
    def manifest_command(
        pipeline: str | None = typer.Option(None, "--pipeline", help=_PIPELINE_OPTION_HELP),
    ) -> int:
        code = _command_manifest(pipeline)
        result_holder.append(code)
        return code


def _register_package_group(app: typer.Typer, result_holder: list[int]) -> None:
    package_app = typer.Typer(no_args_is_help=False)

    @package_app.callback()
    def _package_root() -> None:
        pass

    @package_app.command(
        "build",
        help=(
            "Build a verified Cython extension overlay for the current CPython "
            "ABI and Linux platform. The package itself is not modified."
        ),
    )
    def package_build_command(
        package_root: Path = typer.Option(
            ...,
            "--package-root",
            help="directory containing the installed or source Python package",
        ),
        package_name: str | None = typer.Option(
            None,
            "--package-name",
            help="fully qualified import package (default: package-root directory name)",
        ),
        module_names: list[str] | None = typer.Option(
            None,
            "--module",
            help=(
                "fully qualified implementation module to compile; repeat to select "
                "several (default: every .py except __init__.py and __main__.py)"
            ),
        ),
        jobs: int = typer.Option(
            1, "--jobs", help="parallel Cython and compiler jobs (default: 1)"
        ),
        output_dir: Path = typer.Option(
            ..., "--output-dir", help="new directory to create for the overlay"
        ),
    ) -> int:
        code = _command_package_build(package_root, package_name, module_names, jobs, output_dir)
        result_holder.append(code)
        return code

    @package_app.command("verify", help="verify an overlay and optionally its applied package")
    def package_verify_command(
        overlay_dir: Path = typer.Argument(...),
        target_package_root: Path | None = typer.Option(
            None,
            "--target-package-root",
            help="also verify that this package has the overlay applied",
        ),
    ) -> int:
        code = _command_package_verify(overlay_dir, target_package_root)
        result_holder.append(code)
        return code

    @package_app.command("apply", help="safely apply a verified overlay to an exact package tree")
    def package_apply_command(
        overlay_dir: Path = typer.Argument(...),
        target_package_root: Path = typer.Option(
            ...,
            "--target-package-root",
            help="installed package directory whose matching sources will be replaced",
        ),
    ) -> int:
        code = _command_package_apply(overlay_dir, target_package_root)
        result_holder.append(code)
        return code

    app.add_typer(
        package_app,
        name="package",
        short_help="build, verify, and apply target-bound native runtime overlays",
        help=(
            "Compile Python implementation modules into a runtime-only, target-bound Cython "
            "overlay while preserving the installed wheel's package adapters, "
            "metadata, entry points, and licenses."
        ),
    )


def _register_up_command(app: typer.Typer, result_holder: list[int]) -> None:
    @app.command(
        "up",
        short_help="render the Compose bundle and start the local Airflow runtime",
        help=(
            "Render a self-contained Docker Compose bundle for the pipeline, start it "
            "detached, wait until Airflow reports healthy, and print how to reach it. "
            "The first start pulls images (minutes) and builds the user venv."
        ),
    )
    def up_command(
        pipeline: str | None = typer.Option(None, "--pipeline", help=_PIPELINE_OPTION_HELP),
        data_root: str = typer.Option(
            _environment_data_root(),
            "--data-root",
            help=(
                "host directory mounted at /opt/airflow/data, or a bucket URL "
                "(gs://, s3://, az://) the runtime talks to natively "
                f"(default: $HFLOW_DATA_ROOT, else {PROJECT_CONFIG_FILE_NAME}'s "
                f"data_root, else {DEFAULT_DATA_ROOT})"
            ),
        ),
        bundle_dir: Path | None = typer.Option(
            None,
            "--bundle-dir",
            help="where to render the bundle (default: <data-root>/runtime; ./runtime for bucket URLs)",
        ),
        api_port: int = typer.Option(
            DEFAULT_API_PORT,
            "--api-port",
            help=(
                "host port for the Airflow API, written into a new bundle's .env as "
                f"API_PORT (default: {DEFAULT_API_PORT}, range 1-65535). An existing "
                ".env is never rewritten, so this only takes effect on a bundle that "
                "does not have one yet"
            ),
        ),
        hflow_source: Path | None = typer.Option(
            None,
            "--hflow-source",
            help=(
                "development source checkout to install into the user venv "
                "(default: inferred for editable installs; otherwise the current published version)"
            ),
        ),
        requirements: Path | None = typer.Option(
            None,
            "--requirements",
            help="user requirements file for the task venv (default: hflow only)",
        ),
        pass_env: list[str] = typer.Option(
            [],
            "--pass-env",
            metavar="NAME",
            help=(
                "environment variable to forward into every runtime service without writing its "
                "value to the bundle; repeat for multiple variables"
            ),
        ),
    ) -> int:
        code = _command_up(
            pipeline, data_root, bundle_dir, api_port, hflow_source, requirements, pass_env
        )
        result_holder.append(code)
        return code


def _register_deploy_command(app: typer.Typer, result_holder: list[int]) -> None:
    @app.command(
        "deploy",
        short_help="emit the DAG bundle for an existing Airflow 3 deployment",
        help=(
            "Render the ingest DAG, the user/ files, and a DEPLOY.md with concrete "
            "placement instructions for Astronomer, MWAA, Cloud Composer, and "
            "self-managed Airflow 3. Emits plain files only -- no platform API is called."
        ),
    )
    def deploy_command(
        pipeline: str | None = typer.Option(None, "--pipeline", help=_PIPELINE_OPTION_HELP),
        data_root_uri: str = typer.Option(
            ...,
            "--data-root-uri",
            help="absolute filesystem path or object-store prefix where episode URIs resolve",
        ),
        output_dir: Path = typer.Option(
            DEFAULT_DEPLOY_OUTPUT_DIR,
            "--output-dir",
            help=f"where to write the bundle (default: {DEFAULT_DEPLOY_OUTPUT_DIR})",
        ),
        requirements: Path | None = typer.Option(
            None,
            "--requirements",
            help="user requirements file for the task venv (default: hflow only)",
        ),
        venv_python: str = typer.Option(
            DEFAULT_DEPLOY_VENV_PYTHON,
            "--venv-python",
            help=(
                "the user venv's python interpreter on the workers "
                f"(default: {DEFAULT_DEPLOY_VENV_PYTHON})"
            ),
        ),
        pass_env: list[str] = typer.Option(
            [],
            "--pass-env",
            metavar="NAME",
            help="environment variable the platform must provide to every task; repeat for multiple variables",
        ),
    ) -> int:
        code = _command_deploy(
            pipeline, data_root_uri, output_dir, requirements, venv_python, pass_env
        )
        result_holder.append(code)
        return code


def _register_down_command(app: typer.Typer, result_holder: list[int]) -> None:
    @app.command(
        "down",
        short_help="stop the Compose runtime (containers only; volumes survive)",
        help=(
            "Stop the local Docker Compose runtime rendered by `hflow up`. "
            "Containers are removed while volumes survive unless `--volumes` is "
            "passed for a full reset."
        ),
    )
    def down_command(
        bundle_dir: Path | None = typer.Option(
            None,
            "--bundle-dir",
            help=(
                "the rendered bundle to stop (default: $HFLOW_DATA_ROOT/runtime, "
                f"else {DEFAULT_DATA_ROOT}/runtime; ./runtime for bucket data roots)"
            ),
        ),
        volumes: bool = typer.Option(
            False,
            "--volumes",
            help="also remove the metadata-DB and user-venv volumes (full reset)",
        ),
    ) -> int:
        code = _command_down(bundle_dir, volumes)
        result_holder.append(code)
        return code


def _register_ingest_command(app: typer.Typer, result_holder: list[int]) -> None:
    @app.command(
        "ingest",
        short_help="trigger the master ingest DAG over episode URIs (relative to the data root)",
        help=(
            "Submit one or more episode URIs to the master ingest DAG, or process "
            "them in-process when no runtime is addressed. URIs are relative to the "
            "configured data root; the command prints the triggered run or processing "
            "summary."
        ),
    )
    def ingest_command(
        uris: list[str] = typer.Argument(
            ..., help="episode files, relative to the configured data root"
        ),
        bundle_dir: Path | None = typer.Option(None, "--bundle-dir", help=_bundle_dir_help()),
        airflow_url: str | None = typer.Option(
            None,
            "--airflow-url",
            help=(
                "Airflow API base URL of a remote runtime, e.g. a hosted workspace "
                "(or export HFLOW_AIRFLOW_URL); credentials come from the environment "
                "only: HFLOW_AIRFLOW_TOKEN, or HFLOW_AIRFLOW_USERNAME and "
                "HFLOW_AIRFLOW_PASSWORD"
            ),
        ),
        dag_id: str | None = typer.Option(
            None,
            "--dag-id",
            help="master ingest DAG id on the remote runtime (or export HFLOW_AIRFLOW_DAG_ID)",
        ),
        profile: RunProfile = typer.Option(
            RunProfile.full,
            "--profile",
            help="run profile: which stage sub-DAGs the master enables (default: full)",
        ),
        all_stages: bool = typer.Option(
            False,
            "--all-stages",
            help=(
                "run every stage of --profile on every episode, instead of only the "
                "stages whose steps the catalog does not already record at their "
                "current versions. Use it when an artifact was deleted out from "
                "under a recorded step -- that is the one thing the catalog cannot "
                "see. Applies only when the episodes are processed in this process"
            ),
        ),
        step_names: list[str] | None = typer.Option(
            None,
            "--step",
            metavar="NAME",
            help="run only this registered check, enrichment, or media step; repeat to select more than one",
        ),
        online: bool = typer.Option(
            False,
            "--online",
            help=(
                "latency-first online lane: process the URIs as one immediate batch "
                "(no bin-packing, no stagger) -- for per-episode runs as data lands"
            ),
        ),
        pipeline: str | None = typer.Option(
            None,
            "--pipeline",
            help=(
                "pipeline to run when no runtime is addressed and the episodes are "
                "processed in this process; ignored when a bundle or --airflow-url "
                f"is addressed, since the runtime holds its own copy. Defaults to "
                f"{PROJECT_CONFIG_FILE_NAME}'s `pipeline`, else ./{DEFAULT_PIPELINE_FILE_NAME}"
            ),
        ),
    ) -> int:
        code = _command_ingest(
            uris,
            bundle_dir,
            airflow_url,
            dag_id,
            profile.value,
            all_stages,
            step_names,
            online,
            pipeline,
        )
        result_holder.append(code)
        return code


def _register_status_command(app: typer.Typer, result_holder: list[int]) -> None:
    @app.command(
        "status",
        short_help="runtime health summary with plain-language diagnostics",
        help=(
            "Inspect the health of the local or remote Airflow runtime addressed by "
            "the bundle and endpoint options. Prints a plain-language health summary "
            "and diagnostics for operators."
        ),
    )
    def status_command(
        bundle_dir: Path | None = typer.Option(None, "--bundle-dir", help=_bundle_dir_help()),
        airflow_url: str | None = typer.Option(
            None,
            "--airflow-url",
            help=(
                "Airflow API base URL of a remote runtime, e.g. a hosted workspace "
                "(or export HFLOW_AIRFLOW_URL); credentials come from the environment "
                "only: HFLOW_AIRFLOW_TOKEN, or HFLOW_AIRFLOW_USERNAME and "
                "HFLOW_AIRFLOW_PASSWORD"
            ),
        ),
        dag_id: str | None = typer.Option(
            None,
            "--dag-id",
            help="master ingest DAG id on the remote runtime (or export HFLOW_AIRFLOW_DAG_ID)",
        ),
    ) -> int:
        code = _command_status(bundle_dir, airflow_url, dag_id)
        result_holder.append(code)
        return code


def _register_serve_command(app: typer.Typer, result_holder: list[int]) -> None:
    @app.command(
        "serve",
        short_help=(
            "serve this workspace over HTTP: a REST API over the catalog, and any "
            "UI assets installed (requires the hflow-server package). Distinct from "
            "`up`, which starts the runtime that PROCESSES episodes -- this only "
            "reads the data root, and can trigger a run on a runtime that exists."
        ),
        help=(
            "Serve this workspace over HTTP with REST endpoints over the catalog, "
            "episodes, artifacts, and optional UI assets. The server reads the data "
            "root and can trigger an existing runtime, but it does not process "
            "episodes itself; use `hflow up` for that."
        ),
    )
    def serve_command(
        data_root: str = typer.Option(
            _environment_data_root(),
            "--data-root",
            help=f"workspace data root to browse (default: $HFLOW_DATA_ROOT, else {DEFAULT_DATA_ROOT})",
        ),
        host: str = typer.Option(
            "127.0.0.1",
            "--host",
            help="bind address (default 127.0.0.1; widening past loopback exposes your corpus)",
        ),
        port: int = typer.Option(
            DEFAULT_SERVER_PORT,
            "--port",
            help=f"port to serve on (default {DEFAULT_SERVER_PORT}; auto-retries upward when taken)",
        ),
        no_browser: bool = typer.Option(
            False, "--no-browser", help="do not open a browser after starting (headless use)"
        ),
        read_only: bool = typer.Option(
            False,
            "--read-only",
            help="refuse manifest pinning, saved-query edits, and run triggering",
        ),
        pipeline: str | None = typer.Option(
            None,
            "--pipeline",
            help=(
                "pipeline file for the Pipeline page, optionally with the App variable "
                "name: path/to/pipeline.py[:app]; importing EXECUTES the file, exactly "
                "like `hflow manifest`"
            ),
        ),
    ) -> int:
        code = _command_serve(data_root, host, port, no_browser, read_only, pipeline)
        result_holder.append(code)
        return code


def _build_app(result_holder: list[int]) -> typer.Typer:
    app = typer.Typer(
        name="hflow",
        help="Open-source robotics data pipeline.",
        no_args_is_help=False,
        add_completion=False,
        context_settings={"help_option_names": ["-h", "--help"]},
    )

    def _version_callback(value: bool) -> None:
        if value:
            typer.echo(f"hflow {__version__}")
            raise typer.Exit()

    @app.callback()
    def _root(
        verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable verbose logging"),
        version: bool = typer.Option(
            False,
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the hflow version and exit",
        ),
    ) -> None:
        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format="%(levelname)s %(name)s: %(message)s",
            stream=sys.stderr,
        )

    _register_curate_command(app, result_holder)
    _register_catalog_group(app, result_holder)
    _register_dataset_group(app, result_holder)
    _register_import_group(app, result_holder)
    _register_export_group(app, result_holder)
    _register_verify_group(app, result_holder)
    _register_stale_command(app, result_holder)
    _register_doctor_command(app, result_holder)
    _register_manifest_command(app, result_holder)
    _register_package_group(app, result_holder)
    _register_up_command(app, result_holder)
    _register_deploy_command(app, result_holder)
    _register_down_command(app, result_holder)
    _register_ingest_command(app, result_holder)
    _register_status_command(app, result_holder)
    _register_serve_command(app, result_holder)
    return app


def main(argv: list[str] | None = None) -> int:
    try:
        result_holder: list[int] = []
        app = _build_app(result_holder)
    except ValueError as error:
        # Building the app resolves defaults, which reads hflow.toml. A
        # file that exists and cannot be understood is refused rather than
        # skipped -- falling back to ./data because of a typo would write a
        # corpus into a directory nobody chose -- but it is the user's own
        # just-edited file, so it earns a message and exit 2, not a traceback.
        print(f"hflow: {error}", file=sys.stderr)
        return 2
    click_command = typer.main.get_command(app)
    try:
        # standalone_mode=False so a leaf command's plain `return code`
        # surfaces here as a normal return value (see result_holder below),
        # instead of Click unconditionally exiting 0 on any clean invoke().
        return_value = click_command.main(args=argv, prog_name="hflow", standalone_mode=False)
    except typer.TyperException as error:
        # Every usage error Typer/Click raises (bad choice, missing required
        # argument, unknown subcommand, a bare group with no subcommand)
        # subclasses this. `standalone_mode=False` re-raises it without
        # printing, so the message is shown here before converting to the
        # SystemExit argparse used to raise for the same situations.
        # `.show()` lives on the concrete (vendored) exception classes, not on
        # the public `TyperException` base this except clause is typed to.
        cast(Any, error).show()
        raise SystemExit(error.exit_code) from None
    except typer.Abort:
        raise SystemExit(1) from None
    if result_holder:
        # A leaf command actually ran: report its exit code as a plain
        # return, exactly like the argparse-based `main()` always did.
        return result_holder[-1]
    # No command body ran, so this was --help, --version, or another eager
    # exit -- argparse raised SystemExit for these too.
    raise SystemExit(return_value)


if __name__ == "__main__":
    sys.exit(main())
