"""Render the Docker Compose runtime bundle.

``render_bundle`` writes a self-contained directory a user can inspect, edit,
and run with plain ``docker compose`` -- the SDK provisions the runtime, it
never becomes the runtime (Airflow is not a pip dependency; see
docs/ARCHITECTURE.md "Deployment modes" and references/airflow3-notes.md for
the Airflow 3.3.1 facts this encodes).

Bundle contents:

- ``docker-compose.yaml`` -- the official Airflow 3.3.1 reference compose
  reduced to LocalExecutor: postgres, airflow-init, airflow-apiserver,
  airflow-scheduler, airflow-dag-processor, airflow-triggerer. No redis, no
  celery worker. API bound to ``127.0.0.1:{api_port}`` only. Trap config
  pre-set: dags_are_paused_at_creation=false, load_examples=false,
  object-storage XCom backend (file:// under the data root), DAG bundle list
  pointing at the bundle's ``dags/`` dir.
- ``.env`` -- generated secrets (JWT secret, admin password unless supplied),
  image tags, UID. Regenerating a bundle NEVER overwrites an existing .env
  (create-if-absent: secrets survive re-renders).
- ``dags/`` -- the FIVE generated DAG files (Dyna Figure 4): the master
  ``ingest.py`` plus the four sub-DAGs ``ingest_sync.py`` / ``ingest_meta.py``
  / ``ingest_labels.py`` / ``ingest_media.py`` (see below).
- ``user/`` -- a copy of the user's pipeline file and requirements, mounted
  into the containers.
- ``user-venv`` named volume + an init service that builds the user's venv
  with ``python -m venv`` + ``pip`` inside the Airflow image (the
  external-python pattern: user dependencies never meet Airflow's pins;
  the image ships no ``uv``). The venv is rebuilt
  only when the requirements/hflow-source content hash changes (a marker
  file inside the volume is the checkpoint).

The master DAG (Figure 4's top graph) runs entirely in Airflow's own
environment: a ``@task.branch`` resolves the run profile against a dict
literal baked from :data:`hflow.steps.RUN_PROFILES` at render time, then a
``TriggerDagRunOperator`` per ENABLED stage fires the sub-DAG and waits for it
(deferrable), chained sequentially ``sync >> meta >> labels >> media``.

Each sub-DAG (all task callables run via ``@task.external_python`` against
the user venv; every import lives inside the function bodies -- the operator
extracts them to temp files):

1. ``plan``: ``hflow.batching.plan_batches`` over the conf's ``uris``
   (sizes measured under the mounted data root), returning JSON-serializable
   batches with stagger delays; conf mode ``"online"`` instead returns the
   uris as one immediate batch (latency-first, no stagger).
2. ``process_batch`` (dynamically mapped over batches): sleeps its stagger
   delay, imports the user's pipeline module by file path, calls
   ``app.process(uri, stages={<this sub-DAG's stage>})`` per episode, returns
   counts ``{"processed": n, "quarantined": k, "errors": e}``.
3. The gate: sums the counts and fails the run loudly over the budget of
   ``max(8, ceil(1% of total))``. Checks decide quarantine, so
   the quarantine half of the budget lives ONLY in the meta sub-DAG's
   ``quarantine_budget_gate``; every other stage keeps the
   ``error_budget_gate`` half. Quarantine never deletes; the gate only makes
   mass failure visible.
"""

import hashlib
import logging
import os
import re
import secrets
import shutil
from dataclasses import dataclass
from pathlib import Path

from hflow import __version__
from hflow.runtime._templates import (
    COMPOSE_TEMPLATE,
    DAG_BUNDLE_CONFIG_LIST_JSON,
    MASTER_DAG_TEMPLATE,
    MEDIA_PLAN_FILTER_TEMPLATE,
    SUB_DAG_ERROR_GATE_TEMPLATE,
    SUB_DAG_QUARANTINE_GATE_TEMPLATE,
)
from hflow.steps import RUN_PROFILES, Stage
from hflow.storage import (
    BucketStorageRoot,
    LocalStorageRoot,
    StorageRoot,
    is_bucket_url,
    parse_storage_root,
)

logger = logging.getLogger(__name__)

DEFAULT_AIRFLOW_IMAGE = "apache/airflow:3.3.1"
DEFAULT_POSTGRES_IMAGE = "postgres:16"

# Where a LOCAL host data root is mounted inside every runtime container; the
# user's App must be constructed with exactly this data_root or its outputs
# would land in the container filesystem. Bucket data roots have
# no mount: the App is constructed with the bucket URL itself, and episodes
# spool through the mirror under the containers' XDG_CACHE_HOME (the
# user-venv volume, so downloads persist across restarts).
CONTAINER_DATA_ROOT = "/opt/airflow/data"

# Bucket mode's XCom file store: the bundle-local ./xcom directory mounted
# here (the object-storage XCom backend still wants a file:// path, and with
# no data mount it needs its own).
CONTAINER_XCOM_DIR = "/opt/airflow/xcom-data"

# Where a host Google credentials file is mounted in bucket mode.
CONTAINER_GOOGLE_CREDENTIALS = "/opt/airflow/google-credentials.json"

# Provider credential environment variables passed through into the runtime
# containers for bucket data roots, keyed by credential family. Only
# variables set non-empty at render time are wired: an unconditional
# `${VAR:-}` line would inject empty strings that object_store treats as
# present credentials, breaking the metadata-server/instance-role fallback.
# Values are never written into the bundle -- compose resolves `${VAR}` from
# the shell that runs it. (GOOGLE_APPLICATION_CREDENTIALS and
# GOOGLE_SERVICE_ACCOUNT are host PATHS, so they are handled by mounting the
# file instead of passing the variable through.)
_BUCKET_CREDENTIAL_ENV_VARS: dict[str, tuple[str, ...]] = {
    "s3": (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_ENDPOINT_URL",
        "AWS_ALLOW_HTTP",
    ),
    "gs": ("GOOGLE_SERVICE_ACCOUNT_KEY",),
    "az": (
        "AZURE_STORAGE_ACCOUNT_NAME",
        "AZURE_STORAGE_ACCOUNT_KEY",
        "AZURE_STORAGE_SAS_KEY",
        "AZURE_STORAGE_TOKEN",
    ),
}

# The user venv the Compose bundle's user-venv-init service builds; every DAG
# task's @task.external_python points here. `hflow deploy` renders the same
# DAG against a configurable interpreter path instead (platforms differ).
CONTAINER_VENV_PYTHON = "/opt/venvs/user/bin/python"


@dataclass(frozen=True)
class RuntimeConfig:
    """Everything needed to render a runtime bundle for one pipeline.

    :param pipeline_file: The user's Python file defining the App (copied
        into the bundle; the DAG imports it by path inside the user venv).
    :param app_variable: Name of the App instance in that file.
    :param data_root: A host directory (mounted at ``/opt/airflow/data``
        inside the containers) or a bucket URL (``gs://``/``s3://``/...; no
        mount -- tasks talk to the store natively via the ``[bucket]``
        extra). Episode URIs in ingest conf are resolved against it either
        way.
    :param requirements_file: The user's requirements for the task venv;
        ``None`` means only hflow and its dependencies.
    :param hflow_source: Optional path to an hflow source checkout to install
        into the user venv. When omitted, the runtime installs the exact
        version of the currently running hflow distribution from PyPI.
    :param dag_id: Defaults to ``<pipeline_file stem>_ingest``.
    """

    pipeline_file: Path
    data_root: Path | str
    app_variable: str = "app"
    requirements_file: Path | None = None
    hflow_source: Path | None = None
    dag_id: str | None = None
    airflow_image: str = DEFAULT_AIRFLOW_IMAGE
    postgres_image: str = DEFAULT_POSTGRES_IMAGE
    api_port: int = 8080
    admin_username: str = "airflow"
    admin_password: str | None = None  # None: generated once into .env

    def resolved_dag_id(self) -> str:
        if self.dag_id is not None:
            return self.dag_id
        return f"{self.pipeline_file.stem}_ingest"


@dataclass(frozen=True)
class BundlePaths:
    """Where render_bundle put things.

    ``dag_file`` is the MASTER DAG (``dags/ingest.py``; ``dag_id`` is its id);
    ``sub_dag_files`` are the four stage sub-DAG files in Figure 4 order.
    """

    bundle_dir: Path
    compose_file: Path
    env_file: Path
    dag_file: Path
    user_dir: Path
    api_base_url: str
    admin_username: str
    admin_password: str
    dag_id: str
    sub_dag_files: tuple[Path, ...] = ()


def generate_secret(length_bytes: int = 24) -> str:
    return secrets.token_urlsafe(length_bytes)


def _compose_path_scalar(path: Path) -> str:
    """Escape a host path for a single-quoted scalar in the compose file.

    Two parsers see the value: YAML (embedded single quotes double inside a
    single-quoted scalar) and Docker Compose's own ``${VAR}`` interpolation,
    which runs even inside YAML quotes (``$`` doubles to ``$$``).
    """
    return str(path).replace("$", "$$").replace("'", "''")


def _project_name(bundle_directory: Path) -> str:
    """A unique, stable Compose project name for this bundle.

    Without an explicit top-level ``name:``, Compose defaults to the bundle
    directory's basename -- "runtime" for every default bundle -- and two
    projects on one machine would silently adopt each other's containers and
    share named volumes (`down --volumes` in one would wipe the other's DB).
    """

    directory_digest = hashlib.sha256(str(bundle_directory.resolve()).encode()).hexdigest()[:8]
    return f"hflow-{directory_digest}"


def _google_credentials_file() -> Path | None:
    """The host's Google credentials file to mount, if one exists.

    Precedence: an explicit ``GOOGLE_APPLICATION_CREDENTIALS`` path, then
    object_store's ``GOOGLE_SERVICE_ACCOUNT`` (also a key-file path), then
    gcloud's well-known application-default-credentials location. ``None``
    means mount nothing -- on GCE/GKE the containers reach the instance
    metadata server and need no file at all.
    """
    for variable_name in ("GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_SERVICE_ACCOUNT"):
        explicit = os.environ.get(variable_name)
        if explicit and Path(explicit).is_file():
            return Path(explicit)
    well_known = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    return well_known if well_known.is_file() else None


def _bucket_compose_credentials(bucket_url: str) -> tuple[str, str]:
    """(env suffix, volume-mount suffix) wiring provider credentials through.

    Rendered per the bucket's scheme; re-run ``hflow up`` after changing
    which credential variables your shell exports (the bundle records
    variable NAMES only, never values).
    """
    scheme = bucket_url.split("://", 1)[0].lower()
    credential_family = "az" if scheme in ("az", "abfs", "abfss", "azure") else scheme
    environment_lines: list[str] = []
    mount_suffix = ""
    for variable_name in _BUCKET_CREDENTIAL_ENV_VARS.get(credential_family, ()):
        if os.environ.get(variable_name):
            environment_lines.append(f"\n    {variable_name}: ${{{variable_name}}}")
    if credential_family == "gs":
        credentials_file = _google_credentials_file()
        if credentials_file is not None:
            environment_lines.append(
                f"\n    GOOGLE_APPLICATION_CREDENTIALS: {CONTAINER_GOOGLE_CREDENTIALS}"
            )
            mount_suffix = (
                f"\n    - '{_compose_path_scalar(credentials_file)}:"
                f"{CONTAINER_GOOGLE_CREDENTIALS}:ro'"
            )
    return "".join(environment_lines), mount_suffix


def hflow_distribution_requirement(*, include_bucket_extra: bool) -> str:
    """Return the exact hflow requirement matching the running SDK."""
    distribution_name = "hflow[bucket]" if include_bucket_extra else "hflow"
    return f"{distribution_name}=={__version__}"


def _render_compose(data_root: StorageRoot, hflow_source: Path | None, project_name: str) -> str:
    airflow_hflow_source_mount = ""
    venv_init_hflow_source_mount = ""
    include_bucket_extra = isinstance(data_root, BucketStorageRoot)
    hflow_install_target = hflow_distribution_requirement(include_bucket_extra=include_bucket_extra)
    if hflow_source is not None:
        # Suffix lines appended after the last unconditional volume entry;
        # indentation differs (x-airflow-common vs the user-venv-init service).
        source_scalar = _compose_path_scalar(hflow_source)
        airflow_hflow_source_mount = f"\n    - '{source_scalar}:/opt/hflow-src:ro'"
        venv_init_hflow_source_mount = f"\n      - '{source_scalar}:/opt/hflow-src:ro'"
        hflow_install_target = (
            "/opt/hflow-src[bucket]" if include_bucket_extra else "/opt/hflow-src"
        )
    match data_root:
        case LocalStorageRoot(path=host_data_root):
            data_volume_line = (
                f"\n    - '{_compose_path_scalar(host_data_root)}:{CONTAINER_DATA_ROOT}'"
            )
            xcom_objectstorage_path = f"file://{CONTAINER_DATA_ROOT}/xcom"
            bucket_credentials_env = ""
            bucket_credentials_mount = ""
        case BucketStorageRoot():
            # Episodes never touch the host filesystem in bucket mode: no
            # data mount. XCom still needs a local file store, so the
            # bundle-local ./xcom directory takes that one job over; the task
            # venv gets the [bucket] extra (obstore) to talk to the store.
            data_volume_line = f"\n    - ./xcom:{CONTAINER_XCOM_DIR}"
            xcom_objectstorage_path = f"file://{CONTAINER_XCOM_DIR}"
            bucket_credentials_env, bucket_credentials_mount = _bucket_compose_credentials(
                data_root.url
            )
    return COMPOSE_TEMPLATE.substitute(
        project_name=project_name,
        data_volume_line=data_volume_line,
        xcom_objectstorage_path=xcom_objectstorage_path,
        bucket_credentials_env=bucket_credentials_env,
        bucket_credentials_mount=bucket_credentials_mount,
        hflow_install_target=hflow_install_target,
        dag_bundle_config_list=DAG_BUNDLE_CONFIG_LIST_JSON,
        airflow_hflow_source_mount=airflow_hflow_source_mount,
        venv_init_hflow_source_mount=venv_init_hflow_source_mount,
    )


def _generate_env_values(config: RuntimeConfig) -> dict[str, str]:
    """The ``.env`` contents for a fresh bundle (secrets generated here, once)."""
    return {
        "AIRFLOW_UID": str(os.getuid()),
        "API_PORT": str(config.api_port),
        "AIRFLOW_IMAGE": config.airflow_image,
        "POSTGRES_IMAGE": config.postgres_image,
        "JWT_SECRET": generate_secret(),
        "AIRFLOW_ADMIN_USERNAME": config.admin_username,
        "AIRFLOW_ADMIN_PASSWORD": (
            config.admin_password if config.admin_password is not None else generate_secret()
        ),
    }


def _format_env_file(env_values: dict[str, str]) -> str:
    header = (
        "# Generated by hflow render_bundle. Holds this bundle's secrets and\n"
        "# overridable knobs; NEVER overwritten on re-render (edits survive).\n"
    )
    return header + "".join(f"{key}={value}\n" for key, value in env_values.items())


def _parse_env_file(env_file: Path) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines (comments and blanks skipped, values verbatim)."""
    env_values: dict[str, str] = {}
    for line in env_file.read_text().splitlines():
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith("#") or "=" not in stripped_line:
            continue
        key, _, value = stripped_line.partition("=")
        env_values[key.strip()] = value.strip()
    return env_values


# The master file's own id, not a TriggerDagRunOperator's trigger_dag_id=.
_DAG_ID_PATTERN = re.compile(r'(?<!trigger_)dag_id="([^"]+)"')

# A data_root= keyword with a string literal (optionally Path-wrapped), e.g.
# data_root="./data" or data_root=Path('/mnt/x'). Deliberately narrow: only a
# present, differing literal warns -- variables and expressions stay silent
# (false negatives are fine; the in-container check is authoritative).
_PIPELINE_DATA_ROOT_LITERAL_PATTERN = re.compile(
    r"""data_root\s*=\s*(?:pathlib\s*\.\s*)?(?:Path\(\s*)?(['"])(?P<literal>[^'"\n]*)\1"""
)


def warn_if_pipeline_data_root_differs(
    pipeline_text: str, pipeline_filename: str, expected_data_root: str
) -> None:
    """Warn when a literal pipeline data root differs from the runtime root.

    The authoritative check lives in the generated DAG (the process task
    refuses a wrong ``app.data_root`` at run time); this warning just surfaces
    the mistake minutes earlier, at render time. ``expected_data_root`` is the
    Compose mount point or the deploy bundle's data-root URI.
    """
    for match in _PIPELINE_DATA_ROOT_LITERAL_PATTERN.finditer(pipeline_text):
        literal = match.group("literal")
        # Trailing slashes normalize away everywhere else (parse_storage_root,
        # the run-time guard's str(app.data_root)), so they must not trip the
        # warning either.
        if literal.rstrip("/") != expected_data_root.rstrip("/"):
            logger.warning(
                "%s passes data_root=%r, but inside the runtime the data root must be "
                "%r (where the runtime resolves episode URIs); the ingest DAG will "
                "refuse to process episodes until the pipeline uses that value",
                pipeline_filename,
                literal,
                expected_data_root,
            )


# Figure 4's sub-DAG display names, in the blog figure's own words.
STAGE_TITLES: dict[Stage, str] = {
    Stage.SYNC: "Transform & sync",
    Stage.META: "Metadata",
    Stage.LABELS: "Labels & artifacts",
    Stage.MEDIA: "Media",
}

# One-line stage purposes, shown in the Airflow UI (DAG-list description and
# each sub-DAG's doc_md). One owner for the demo-facing vocabulary.
STAGE_DESCRIPTIONS: dict[Stage, str] = {
    Stage.SYNC: "Canonical transform -- the critical path (Figure 4: Transform & sync).",
    Stage.META: "Quality checks + catalog registration, with the run's quarantine budget.",
    Stage.LABELS: "Enrichments -- non-critical, failure isolated.",
    Stage.MEDIA: "Derived media: per-camera contact sheets recorded as catalog artifacts.",
}


def _pipeline_stem(master_dag_id: str) -> str:
    """The pipeline's short name, shared by sub-DAG ids, UI tags, and display
    names: the master id minus its historical ``_ingest`` suffix."""
    return master_dag_id.removesuffix("_ingest") or master_dag_id


def sub_dag_id_for_stage(master_dag_id: str, stage: Stage) -> str:
    """The stage sub-DAG's id, derived from the master's id.

    The master keeps the historical ``<stem>_ingest`` id (the CLI,
    ``load_bundle``, and the docs all address it); each sub-DAG replaces the
    ``_ingest`` suffix with its stage name: ``<stem>_sync`` etc.
    """
    return f"{_pipeline_stem(master_dag_id)}_{stage.value}"


def bundle_dag_ids(master_dag_id: str) -> list[str]:
    """All five generated DAG ids: the master first, then Figure 4's stages."""
    return [
        master_dag_id,
        *(sub_dag_id_for_stage(master_dag_id, stage) for stage in Stage),
    ]


def _validate_dag_identifiers(dag_id: str, app_variable: str | None = None) -> None:
    """Refuse anything that could break (or smuggle code into) a DAG file."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", dag_id):
        raise ValueError(f"dag_id {dag_id!r} may only contain [A-Za-z0-9_.-]")
    if app_variable is not None and not app_variable.isidentifier():
        raise ValueError(f"app_variable {app_variable!r} is not a Python identifier")


def _run_profiles_literal() -> str:
    """:data:`hflow.steps.RUN_PROFILES` as a Python dict literal.

    Baked into the master DAG at render time so the master runs without
    importing hflow (it executes in Airflow's own environment); steps.py
    stays the one owner of the vocabulary -- the baked copy is generated code
    refreshed on every re-render. Stage names keep declaration order (Figure
    4's left-to-right order).
    """
    profile_lines = ["{"]
    for profile_name, profile_stages in RUN_PROFILES.items():
        ordered_stage_names = [stage.value for stage in Stage if stage in profile_stages]
        rendered_stages = ", ".join(f'"{stage_name}"' for stage_name in ordered_stage_names)
        single_element_comma = "," if len(ordered_stage_names) == 1 else ""
        profile_lines.append(f'    "{profile_name}": ({rendered_stages}{single_element_comma}),')
    profile_lines.append("}")
    return "\n".join(profile_lines)


def _profile_table_markdown_rows() -> str:
    """``RUN_PROFILES`` as markdown table rows for the master's ``doc_md``.

    Baked at render time like the profile literal, so the UI documentation
    can never drift from the vocabulary's one owner in steps.py.
    """
    rows = []
    for profile_name, profile_stages in RUN_PROFILES.items():
        ordered = ", ".join(stage.value for stage in Stage if stage in profile_stages)
        rows.append(f"| `{profile_name}` | {ordered} |")
    return "\n".join(rows)


def render_master_dag_source(*, dag_id: str) -> str:
    """The master DAG's Python source (Figure 4's top graph).

    Needs no data root or venv: it runs entirely in Airflow's environment and
    only resolves the profile and triggers sub-DAGs.
    """
    _validate_dag_identifiers(dag_id)
    pipeline_stem = _pipeline_stem(dag_id)
    return MASTER_DAG_TEMPLATE.substitute(
        dag_id=dag_id,
        run_profiles_literal=_run_profiles_literal(),
        profile_table_rows=_profile_table_markdown_rows(),
        pipeline_tag=pipeline_stem,
        master_display_name=f"{pipeline_stem} · ingest (master)",
        sync_dag_id=sub_dag_id_for_stage(dag_id, Stage.SYNC),
        meta_dag_id=sub_dag_id_for_stage(dag_id, Stage.META),
        labels_dag_id=sub_dag_id_for_stage(dag_id, Stage.LABELS),
        media_dag_id=sub_dag_id_for_stage(dag_id, Stage.MEDIA),
    )


def render_sub_dag_source(
    *,
    master_dag_id: str,
    stage: Stage,
    pipeline_filename: str,
    app_variable: str,
    data_root: str,
    venv_python: str,
) -> str:
    """One stage sub-DAG's Python source, shared by Compose and deploy rendering.

    ``data_root`` and ``venv_python`` are the two facts the deployment modes
    disagree on (Compose: :data:`CONTAINER_DATA_ROOT` /
    :data:`CONTAINER_VENV_PYTHON`; deploy: the user's URI and platform venv).
    """
    sub_dag_id = sub_dag_id_for_stage(master_dag_id, stage)
    _validate_dag_identifiers(master_dag_id, app_variable)
    _validate_dag_identifiers(sub_dag_id)
    match stage:
        case Stage.META:
            template, gate_name = SUB_DAG_QUARANTINE_GATE_TEMPLATE, "quarantine_budget_gate"
        case Stage.SYNC | Stage.LABELS | Stage.MEDIA:
            template, gate_name = SUB_DAG_ERROR_GATE_TEMPLATE, "error_budget_gate"
    pipeline_stem = _pipeline_stem(master_dag_id)
    # $data_root, $venv_python, and $pipeline_filename are substituted as
    # repr() literals, never as bare text inside a pre-quoted template
    # slot: a raw platform path (Windows venv interpreters use backslashes)
    # or a data root containing a quote character would otherwise either
    # break the generated file's Python syntax or -- worse -- close the
    # surrounding string literal early and splice arbitrary text into the
    # generated DAG's source (#44). repr() on a str always yields a
    # self-escaping literal, so this holds for any input, not just the
    # platform-path case that surfaced it.
    data_root_literal = repr(data_root)
    venv_python_literal = repr(venv_python)
    pipeline_filename_literal = repr(pipeline_filename)
    # Substituted separately because Template.substitute never re-expands
    # variables inside substituted VALUES -- the filter's own $data_root must
    # be resolved before injection. Local roots only: probing a BUCKET
    # episode's channel list would download the whole file at plan time,
    # costing more than the skipped camera-less cycle saves.
    stage_plan_filter = (
        MEDIA_PLAN_FILTER_TEMPLATE.substitute(data_root=data_root_literal)
        if stage is Stage.MEDIA and not is_bucket_url(data_root)
        else ""
    )
    return template.substitute(
        dag_id=sub_dag_id,
        master_dag_id=master_dag_id,
        stage_title=STAGE_TITLES[stage],
        stage_name=stage.value,
        stage_description=STAGE_DESCRIPTIONS[stage],
        pipeline_tag=pipeline_stem,
        sub_display_name=f"{pipeline_stem} · {stage.value}",
        gate_name=gate_name,
        pipeline_filename=pipeline_filename_literal,
        app_variable=app_variable,
        data_root=data_root_literal,
        venv_python=venv_python_literal,
        stage_plan_filter=stage_plan_filter,
    )


def render_dag_sources(
    *,
    master_dag_id: str,
    pipeline_filename: str,
    app_variable: str,
    data_root: str,
    venv_python: str,
) -> dict[str, str]:
    """dag_id -> source for all five DAGs (master first, then Figure 4 order)."""
    sources = {master_dag_id: render_master_dag_source(dag_id=master_dag_id)}
    for stage in Stage:
        sources[sub_dag_id_for_stage(master_dag_id, stage)] = render_sub_dag_source(
            master_dag_id=master_dag_id,
            stage=stage,
            pipeline_filename=pipeline_filename,
            app_variable=app_variable,
            data_root=data_root,
            venv_python=venv_python,
        )
    return sources


def infer_hflow_source() -> Path | None:
    """The source checkout the imported ``hflow`` package lives in, if any.

    An editable/source install can supply its own checkout as the development
    default. Returns ``None`` for a site-packages wheel install, including a
    virtual environment created inside an hflow checkout, which makes the
    runtime install the same published distribution version instead.
    """
    import hflow

    package_dir = Path(hflow.__file__).resolve().parent
    for ancestor in package_dir.parents:
        pyproject = ancestor / "pyproject.toml"
        source_package_directories = (ancestor / "src" / "hflow", ancestor / "hflow")
        imported_from_checkout = any(
            source_package_directory.resolve() == package_dir
            for source_package_directory in source_package_directories
        )
        if (
            imported_from_checkout
            and pyproject.is_file()
            and 'name = "hflow"' in pyproject.read_text()
        ):
            return ancestor
    return None


def load_bundle(bundle_dir: Path | str) -> BundlePaths:
    """Reconstruct :class:`BundlePaths` for an already-rendered bundle.

    The CLI's ``down``/``ingest``/``status`` take only a bundle directory;
    everything else (port, credentials, dag id) is read back from the bundle's
    own files -- the .env for secrets, the generated DAG for its id.
    """
    bundle_directory = Path(bundle_dir)
    compose_file = bundle_directory / "docker-compose.yaml"
    env_file = bundle_directory / ".env"
    dag_file = bundle_directory / "dags" / "ingest.py"
    for required_file in (compose_file, env_file, dag_file):
        if not required_file.is_file():
            raise FileNotFoundError(
                f"no rendered bundle at {bundle_directory} (missing {required_file.name}); "
                "run `hflow up` (or render_bundle) first"
            )
    dag_id_match = _DAG_ID_PATTERN.search(dag_file.read_text())
    if dag_id_match is None:
        raise ValueError(f"{dag_file} has no dag_id=... -- not an hflow-generated DAG")
    env_values = _parse_env_file(env_file)
    return BundlePaths(
        bundle_dir=bundle_directory,
        compose_file=compose_file,
        env_file=env_file,
        dag_file=dag_file,
        user_dir=bundle_directory / "user",
        sub_dag_files=tuple(
            bundle_directory / "dags" / f"ingest_{stage.value}.py" for stage in Stage
        ),
        api_base_url=f"http://127.0.0.1:{env_values.get('API_PORT', '8080')}",
        admin_username=env_values.get("AIRFLOW_ADMIN_USERNAME", "airflow"),
        admin_password=env_values.get("AIRFLOW_ADMIN_PASSWORD", ""),
        dag_id=dag_id_match.group(1),
    )


def render_bundle(config: RuntimeConfig, bundle_dir: Path | str) -> BundlePaths:
    """Write (or refresh) the bundle at ``bundle_dir``.

    Idempotent and re-render-safe: generated files are overwritten (they are
    derived from config), but ``.env`` secrets are created once and kept.
    """
    bundle_directory = Path(bundle_dir)
    pipeline_source = Path(config.pipeline_file)
    if not pipeline_source.is_file():
        raise FileNotFoundError(pipeline_source)

    dags_dir = bundle_directory / "dags"
    logs_dir = bundle_directory / "logs"
    user_dir = bundle_directory / "user"
    for directory in (dags_dir, logs_dir, user_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # Pre-create the bind-mount source directories: docker would otherwise
    # create missing ones owned by root. Local mode: xcom/ under the data
    # root backs the object-storage XCom path. Bucket mode: no data mount at
    # all -- the bundle-local xcom/ takes that job, episodes live in the
    # bucket, and the DAGs are rendered against the bucket URL itself.
    parsed_data_root = parse_storage_root(config.data_root)
    match parsed_data_root:
        case LocalStorageRoot(path=raw_local_root):
            resolved_data_root: StorageRoot = LocalStorageRoot(
                raw_local_root.expanduser().resolve()
            )
            (resolved_data_root.path / "xcom").mkdir(parents=True, exist_ok=True)
            dag_data_root = CONTAINER_DATA_ROOT
        case BucketStorageRoot():
            resolved_data_root = parsed_data_root
            (bundle_directory / "xcom").mkdir(exist_ok=True)
            dag_data_root = parsed_data_root.url

    # user/ contents are derived from config, so they are always refreshed.
    shutil.copyfile(pipeline_source, user_dir / pipeline_source.name)
    warn_if_pipeline_data_root_differs(
        (user_dir / pipeline_source.name).read_text(), pipeline_source.name, dag_data_root
    )
    if config.requirements_file is not None:
        requirements_source = Path(config.requirements_file)
        if not requirements_source.is_file():
            raise FileNotFoundError(requirements_source)
        shutil.copyfile(requirements_source, user_dir / "requirements.txt")

    hflow_source = (
        Path(config.hflow_source).expanduser().resolve()
        if config.hflow_source is not None
        else None
    )
    compose_file = bundle_directory / "docker-compose.yaml"
    compose_file.write_text(
        _render_compose(resolved_data_root, hflow_source, _project_name(bundle_directory))
    )

    # Five DAG files (Figure 4): the master keeps the historical ingest.py
    # filename (load_bundle reads the dag id back from it); each sub-DAG gets
    # its stage-suffixed sibling.
    master_dag_id = config.resolved_dag_id()
    dag_sources = render_dag_sources(
        master_dag_id=master_dag_id,
        pipeline_filename=pipeline_source.name,
        app_variable=config.app_variable,
        data_root=dag_data_root,
        venv_python=CONTAINER_VENV_PYTHON,
    )
    dag_file = dags_dir / "ingest.py"
    dag_file.write_text(dag_sources[master_dag_id])
    sub_dag_files: list[Path] = []
    for stage in Stage:
        sub_dag_file = dags_dir / f"ingest_{stage.value}.py"
        sub_dag_file.write_text(dag_sources[sub_dag_id_for_stage(master_dag_id, stage)])
        sub_dag_files.append(sub_dag_file)

    # Create-if-absent: an existing .env is never rewritten, so its secrets
    # (and any user edits) survive every re-render. It holds the JWT secret
    # and admin password, so it is owner-only (0600) -- created atomically,
    # and healed on re-render for bundles written by older versions.
    env_file = bundle_directory / ".env"
    try:
        env_descriptor = os.open(env_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        env_file.chmod(0o600)
    else:
        with os.fdopen(env_descriptor, "w") as env_stream:
            env_stream.write(_format_env_file(_generate_env_values(config)))

    # Report what is actually in effect: the .env on disk wins over config
    # (a re-render with a changed port does not change a preserved .env).
    effective_env = _parse_env_file(env_file)
    effective_api_port = effective_env.get("API_PORT", str(config.api_port))
    return BundlePaths(
        bundle_dir=bundle_directory,
        compose_file=compose_file,
        env_file=env_file,
        dag_file=dag_file,
        user_dir=user_dir,
        api_base_url=f"http://127.0.0.1:{effective_api_port}",
        admin_username=effective_env.get("AIRFLOW_ADMIN_USERNAME", config.admin_username),
        admin_password=effective_env.get("AIRFLOW_ADMIN_PASSWORD", config.admin_password or ""),
        dag_id=master_dag_id,
        sub_dag_files=tuple(sub_dag_files),
    )
