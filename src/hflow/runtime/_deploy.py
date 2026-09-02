"""``hflow deploy``: emit the DAG bundle for an existing Airflow 3 deployment.

Docker mode (``_bundle``) provisions a runtime; this module targets one the
user already operates -- Astronomer, MWAA, Cloud Composer, or self-managed
(docs/ARCHITECTURE.md "Deployment modes", mode 2). ``render_deploy_bundle``
writes plain files and instructions; it makes no API call to any platform.

Output directory contents:

- ``dags/<dag_id>.py`` plus four ``dags/<sub_dag_id>.py`` -- the same five
  ingest DAGs the Compose runtime generates (ONE set of templates in
  ``_templates``): the ingest stage graph's master and its
  sync/meta/labels/media sub-DAGs, rendered against the deployment's data
  root and user-venv
  interpreter instead of the Compose container values -- plus a
  ``dags/.airflowignore`` so platforms that sync ``user/`` inside the dags
  folder never parse the pipeline file in Airflow's own environment.
- ``user/`` -- a copy of the pipeline file (and ``requirements.txt`` when
  supplied), to be placed wherever the platform allows; the DAG finds it via
  the ``HFLOW_USER_DIR`` environment variable (default ``/opt/user``).
- ``DEPLOY.md`` -- concrete instructions: where dags/ and user/ go per
  platform, the ``AIRFLOW__DAG_PROCESSOR__DAG_BUNDLE_CONFIG_LIST`` JSON for
  self-managed deployments (references/airflow3-notes.md, "DAG bundles"), the
  external-python venv's package list, and the environment the DAG expects.

The data root is either an absolute filesystem path every worker sees or a
supported object-store prefix. Object roots use hflow's obstore-backed
spool-through-mirror boundary; workers therefore keep the processing core on
local files without pretending bucket keys have ``pathlib.Path`` semantics.
"""

import errno
import os
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path

from hflow.app import DEFAULT_APP_VARIABLE
from hflow.runtime._bundle import (
    BundleKind,
    copy_user_project,
    hflow_distribution_requirement,
    parse_environment_variable_names,
    render_dag_sources,
    sub_dag_id_for_stage,
    warn_if_pipeline_data_root_differs,
    write_bundle_manifest,
)
from hflow.runtime._templates import (
    DAG_BUNDLE_CONFIG_LIST_JSON,
    EXTERNAL_PYTHON_BOOTSTRAP_REQUIREMENT,
)
from hflow.stage_execution import USER_DIRECTORY_DEFAULT
from hflow.steps import RUN_PROFILES, Stage
from hflow.storage import is_bucket_url, parse_storage_root

# The interpreter of the user venv on the deployment's workers. Matches the
# Compose runtime's convention so a venv built from these instructions is
# drop-in there too; every platform can override it (--venv-python).
DEFAULT_DEPLOY_VENV_PYTHON = "/opt/venvs/user/bin/python"

# What the external-python venv must contain besides the user's requirements
# (references/airflow3-notes.md, "External-python": virtualenv preinstalled;
# pendulum + lazy_object_proxy for the operator's datetime context probes).
# One owner with the Compose renderer, and pinned: an unpinned pendulum here
# against a pinned one there is a task that fails its bootstrap probe on one
# deployment path only. virtualenv and lazy_object_proxy are deliberately
# gone: with expect_airflow=False the venv probe is exactly `import pendulum`,
# and proxies resolve host-side before pickling so they never cross into it.
DEPLOY_VENV_BASE_PACKAGES = (EXTERNAL_PYTHON_BOOTSTRAP_REQUIREMENT,)


def validate_data_root_uri(data_root_uri: str) -> str:
    """Parse a deploy data root into a normalized path or bucket prefix.

    Relative paths are refused because they cannot resolve consistently on
    remote workers. Supported bucket URLs are parsed by the same boundary the
    application uses; unsupported schemes fail loudly.
    """
    if "://" in data_root_uri:
        try:
            storage_root = parse_storage_root(data_root_uri)
        except ValueError as error:
            raise ValueError(f"invalid data root {data_root_uri!r}: {error}") from error
        if is_bucket_url(data_root_uri):
            return str(storage_root)
        raise ValueError(f"data root {data_root_uri!r} is not a supported object-store URL")
    normalized_path = data_root_uri.rstrip("/")
    if data_root_uri.startswith("/") and normalized_path:
        return normalized_path
    raise ValueError(
        f"data root {data_root_uri!r} is not an absolute filesystem path -- a deploy "
        "bundle runs on remote workers, so relative paths cannot resolve consistently"
    )


@dataclass(frozen=True)
class DeployConfig:
    """Everything needed to emit a deploy bundle for one pipeline.

    :param pipeline_file: The user's Python file defining the App (copied into
        the bundle's ``user/``; the DAG imports it by path inside the venv).
    :param data_root_uri: Absolute mounted path or supported object-store
        prefix under which episode URIs resolve (validated loudly).
    :param app_variable: Name of the App instance in the pipeline file.
    :param requirements_file: The user's requirements for the task venv;
        ``None`` means only the base packages.
    :param venv_python_path: The user venv's interpreter on the workers
        (platforms differ; the DAG's ``@task.external_python`` points here).
    :param dag_id: Defaults to ``<pipeline_file stem>_ingest`` (the same rule
        as the Compose runtime's :class:`~hflow.runtime.RuntimeConfig`).
    :param task_queue: Optional Airflow queue name stamped onto every stage
        task -- routes this pipeline's tasks to a dedicated worker pool on
        deployments running more than one (per-team or per-workspace
        workers). ``None`` keeps the executor's default queue.
    :param passthrough_environment_variables: Explicit names the platform
        operator must set in every task environment. Values are never written
        to the generated bundle.
    """

    pipeline_file: Path
    data_root_uri: str
    app_variable: str = DEFAULT_APP_VARIABLE
    requirements_file: Path | None = None
    venv_python_path: str = DEFAULT_DEPLOY_VENV_PYTHON
    dag_id: str | None = None
    task_queue: str | None = None
    passthrough_environment_variables: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Parse at the boundary: an invalid data root never becomes a config.
        # The normalized form (no trailing slash) replaces the raw input so
        # every later join has exactly one owner of that fact.
        object.__setattr__(self, "data_root_uri", validate_data_root_uri(self.data_root_uri))
        object.__setattr__(
            self,
            "passthrough_environment_variables",
            parse_environment_variable_names(self.passthrough_environment_variables),
        )

    def resolved_dag_id(self) -> str:
        if self.dag_id is not None:
            return self.dag_id
        return f"{Path(self.pipeline_file).stem}_ingest"


@dataclass(frozen=True)
class DeployPaths:
    """Where render_deploy_bundle put things.

    ``dag_file`` is the MASTER DAG (``dag_id`` is its id); ``sub_dag_files``
    are the four stage sub-DAG files in stage-graph order (declaration order).
    """

    output_dir: Path
    dag_file: Path
    user_dir: Path
    deploy_md: Path
    dag_id: str
    sub_dag_files: tuple[Path, ...] = ()


def render_deploy_bundle(config: DeployConfig, output_dir: Path | str) -> DeployPaths:
    """Write the deploy bundle at ``output_dir`` (overwriting generated files).

    Everything emitted is derived from ``config``, so re-rendering is always
    safe -- there is no equivalent of the Compose bundle's preserved ``.env``.
    """
    output_directory = Path(output_dir)
    pipeline_source = Path(config.pipeline_file)
    if pipeline_source.is_dir():
        # Deliberately FileNotFoundError and not IsADirectoryError: the accurate
        # class subclasses OSError, not FileNotFoundError, so an `except
        # FileNotFoundError` stops catching it. Two such handlers sit in this
        # call path, cli.py:560 for `up` and cli.py:601 for `deploy`, and both
        # turn this into exit 2; thirteen more name it elsewhere under
        # src/hflow/, and callers outside the repo are the real unknown.
        # The errno carries the accurate text without moving the class. #100.
        raise FileNotFoundError(errno.EISDIR, os.strerror(errno.EISDIR), str(pipeline_source))
    if not pipeline_source.is_file():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(pipeline_source))

    dags_dir = output_directory / "dags"
    user_dir = output_directory / "user"
    for directory in (dags_dir, user_dir):
        directory.mkdir(parents=True, exist_ok=True)

    copy_user_project(
        pipeline_source,
        user_dir,
        bundle_directory=output_directory,
        data_root=config.data_root_uri,
    )
    warn_if_pipeline_data_root_differs(
        (user_dir / pipeline_source.name).read_text(), pipeline_source.name, config.data_root_uri
    )
    requirements_copied = False
    if config.requirements_file is not None:
        requirements_source = Path(config.requirements_file)
        if requirements_source.is_dir():
            # FileNotFoundError, not IsADirectoryError, for the reason above.
            raise FileNotFoundError(
                errno.EISDIR, os.strerror(errno.EISDIR), str(requirements_source)
            )
        if not requirements_source.is_file():
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), str(requirements_source)
            )
        shutil.copyfile(requirements_source, user_dir / "requirements.txt")
        requirements_copied = True

    # On MWAA/Cloud Composer user/ can only ride along inside the synced dags
    # folder, where the parser would try to import the pipeline inside
    # AIRFLOW'S environment (hflow lives only in the task venv, so that
    # import fails noisily). "user/" matches under both .airflowignore
    # syntaxes (regexp and glob), so ship the guard unconditionally.
    (dags_dir / ".airflowignore").write_text("user/\n")

    # Five DAG files (the stage graph), each named after its dag id -- platforms
    # sync whole dags/ folders, so generic filenames would collide across
    # pipelines.
    dag_id = config.resolved_dag_id()
    dag_sources = render_dag_sources(
        master_dag_id=dag_id,
        pipeline_filename=pipeline_source.name,
        app_variable=config.app_variable,
        data_root=config.data_root_uri,
        venv_python=config.venv_python_path,
        task_queue=config.task_queue,
    )
    dag_file = dags_dir / f"{dag_id}.py"
    dag_file.write_text(dag_sources[dag_id])
    sub_dag_files: list[Path] = []
    for stage in Stage:
        sub_dag_id = sub_dag_id_for_stage(dag_id, stage)
        sub_dag_file = dags_dir / f"{sub_dag_id}.py"
        sub_dag_file.write_text(dag_sources[sub_dag_id])
        sub_dag_files.append(sub_dag_file)

    write_bundle_manifest(
        output_directory,
        kind=BundleKind.DEPLOY,
        dag_id=dag_id,
        data_root=config.data_root_uri,
        app_variable=config.app_variable,
        pipeline_filename=pipeline_source.name,
        requirements_included=requirements_copied,
        task_queue=config.task_queue,
        venv_python=config.venv_python_path,
        passthrough_environment_variables=config.passthrough_environment_variables,
    )

    deploy_md = output_directory / "DEPLOY.md"
    deploy_md.write_text(
        _render_deploy_md(config, dag_id=dag_id, requirements_copied=requirements_copied)
    )
    return DeployPaths(
        output_dir=output_directory,
        dag_file=dag_file,
        user_dir=user_dir,
        deploy_md=deploy_md,
        dag_id=dag_id,
        sub_dag_files=tuple(sub_dag_files),
    )


def _render_deploy_md(config: DeployConfig, *, dag_id: str, requirements_copied: bool) -> str:
    """The bundle's instructions -- everything a platform operator must do.

    Kept as one generated document (not a static file) so the concrete values
    (dag id, data root, interpreter path, requirements) are already filled in.
    """
    dag_filename = f"{dag_id}.py"
    sub_dag_ids = [sub_dag_id_for_stage(dag_id, stage) for stage in Stage]
    sub_dag_list = ", ".join(f"`{sub_dag_id}.py`" for sub_dag_id in sub_dag_ids)
    profile_names = " | ".join(f'"{profile_name}"' for profile_name in RUN_PROFILES)
    pipeline_filename = Path(config.pipeline_file).name
    # shlex.quote protects the bucket extra's brackets from shells such as zsh.
    hflow_requirement = hflow_distribution_requirement(
        include_bucket_extra=is_bucket_url(config.data_root_uri)
    )
    venv_packages = " ".join((*DEPLOY_VENV_BASE_PACKAGES, shlex.quote(hflow_requirement)))
    requirements_line = (
        f"{config.venv_python_path.removesuffix('/python')}/pip install -r user/requirements.txt"
        if requirements_copied
        else "# (no user requirements file was supplied)"
    )
    passthrough_environment_rows = "".join(
        f"| `{variable_name}` | pipeline-defined | required by this pipeline; set on every "
        "worker's task environment |\n"
        for variable_name in config.passthrough_environment_variables
    )
    return f"""\
# Deploying `{dag_id}` to your Airflow 3 deployment

Generated by `hflow deploy` -- re-render instead of editing. No platform
API is called; you place these files with the tools you already use.

## What is in this bundle

- `dags/{dag_filename}` -- the master ingest DAG: it resolves
  the run profile and triggers only the enabled stage sub-DAGs, chained
  sequentially. Trigger conf: `{{"uris": [...], "profile": {profile_names},
  "mode": "batch"|"online", "batch_count": optional int}}`; URIs resolve
  under the data root below. It runs entirely in Airflow's own environment
  and needs `apache-airflow-providers-standard` (bundled with the official
  image and every managed platform) plus a running triggerer (its
  waits are deferrable).
- {sub_dag_list} -- the four stage sub-DAGs the
  master triggers (transform & sync, metadata, labels & artifacts, media).
  Their tasks run in the task venv below.
- `user/{pipeline_filename}` -- your pipeline, imported by the sub-DAGs' tasks.
- `user/requirements.txt` -- your task-venv requirements{"" if requirements_copied else " (not supplied)"}.

## 1. Build the task venv on every worker

Every sub-DAG task runs through `@task.external_python` against
`{config.venv_python_path}`. On every machine that executes tasks, create a
venv at that path and install the base packages plus your requirements
(`virtualenv` must be preinstalled in the venv, and `pendulum` +
`lazy_object_proxy` satisfy the operator's datetime-context probes -- see the
Airflow external-python docs):

```bash
python -m venv {config.venv_python_path.removesuffix("/bin/python")}
{config.venv_python_path.removesuffix("/python")}/pip install {venv_packages}
{requirements_line}
```

If your platform cannot host a venv at that path (managed images differ),
create it wherever the platform allows and re-render with
`--venv-python <interpreter path>`.

## 2. Place `dags/` and `user/`

The DAG loads your pipeline from `$HFLOW_USER_DIR/{pipeline_filename}`
(default `{USER_DIRECTORY_DEFAULT}`), so set `HFLOW_USER_DIR` on all Airflow
components to wherever `user/` lands.

### Astronomer

- Copy the five DAG files in `dags/` into your Astro project's `dags/` folder.
- Copy `user/` into the project's `include/user/` (deployed at
  `/usr/local/airflow/include/user`).
- Set `HFLOW_USER_DIR=/usr/local/airflow/include/user` in the Deployment's
  environment variables, then `astro deploy`.

### Amazon MWAA

- Upload the contents of `dags/` (the five DAG files plus `.airflowignore`)
  to `s3://<your-mwaa-bucket>/dags/`.
- Upload `user/` to `s3://<your-mwaa-bucket>/dags/user/` (MWAA syncs the dags
  prefix to `/usr/local/airflow/dags`; the bundled `.airflowignore` keeps the
  parser away from your pipeline file, which imports task-venv-only packages).
- Set `HFLOW_USER_DIR=/usr/local/airflow/dags/user` (MWAA environment
  configuration, "Environment variables").

### Cloud Composer

- Upload the contents of `dags/` (the five DAG files plus `.airflowignore`)
  to `gs://<your-composer-bucket>/dags/`.
- Upload `user/` to `gs://<your-composer-bucket>/dags/user/` (mounted at
  `/home/airflow/gcs/dags/user`; the bundled `.airflowignore` keeps the
  parser away from your pipeline file, which imports task-venv-only packages).
- Set `HFLOW_USER_DIR=/home/airflow/gcs/dags/user` (Composer environment
  variables).

### Self-managed Airflow 3

- Place the five `dags/*.py` files in a directory readable by the dag
  processor, e.g. `/opt/airflow/dags`, and point Airflow 3's DAG-bundle list
  at it:

```bash
AIRFLOW__DAG_PROCESSOR__DAG_BUNDLE_CONFIG_LIST='{DAG_BUNDLE_CONFIG_LIST_JSON}'
```

- Place `user/` at `{USER_DIRECTORY_DEFAULT}` on every worker (or anywhere, plus
  `HFLOW_USER_DIR`).

## 3. Environment the DAG expects

| variable | value | why |
| --- | --- | --- |
| `HFLOW_USER_DIR` | absolute path of `user/` on the workers | where the tasks import `{pipeline_filename}` from (default `{USER_DIRECTORY_DEFAULT}`) |
{passthrough_environment_rows}

Data root: episode URIs in the trigger conf resolve under
`{config.data_root_uri}`. Construct your pipeline's App WITHOUT a
`data_root` argument (recommended: the task exports `HFLOW_DATA_ROOT`
before importing your pipeline, so the same file also runs unedited in the
dev loop), or with exactly `data_root="{config.data_root_uri}"` -- the DAG
refuses to process episodes under any other root.

Object-store roots use the optional obstore backend in the task venv and the
provider's standard worker credentials. Each worker spools inputs and outputs
through an etag-validated local mirror (set `HFLOW_MIRROR_DIR` to place it
on suitable ephemeral storage). Absolute paths must be mounted identically on
every worker.

## 4. Trigger an ingest

Trigger the MASTER DAG `{dag_id}`; it fans out to the stage sub-DAGs its
profile enables (run profiles: {profile_names}; `"mode": "online"` is the
latency-first per-episode lane -- one immediate batch, no stagger).

Airflow UI: trigger `{dag_id}` with conf
`{{"uris": ["episodes-in/run_0001.mcap"], "profile": "full", "mode": "batch"}}`.
REST: `POST /api/v2/dags/{dag_id}/dagRuns` with
`{{"logical_date": null, "conf": {{"uris": [...], "profile": "full",
"mode": "batch"}}}}` (Bearer token from `POST /auth/token`).
"""
