"""Runtime lifecycle: render + ``compose up`` + health wait, and diagnostics.

The pieces ``hflow up``/``status`` and ``App.run()`` share, kept in the
library so no behavior lives only in the CLI (cli.py's own rule).
"""

import time
from collections.abc import Callable, Sequence
from pathlib import Path

from hflow.runtime._bundle import BundlePaths, RuntimeConfig, bundle_dag_ids, render_bundle
from hflow.runtime._client import AirflowClient, AirflowClientError, AirflowHealth
from hflow.runtime._compose import ComposeError, compose_ps, compose_up_detached

# How often the health wait narrates its latest status: rare enough not to
# scroll the terminal, frequent enough to prove the wait is alive.
_HEALTH_PROGRESS_INTERVAL_S = 15.0


def client_for_bundle(paths: BundlePaths) -> AirflowClient:
    """An :class:`AirflowClient` speaking to this bundle's api-server."""
    return AirflowClient(paths.api_base_url, paths.admin_username, paths.admin_password)


def _throttled_health_progress(on_progress: Callable[[str], None]) -> Callable[[str], None]:
    """Forward at most one health summary to ``on_progress`` per interval.

    ``wait_until_healthy`` polls every few seconds; narrating every poll would
    drown the terminal, so this adapter drops all but a ~15s heartbeat.
    """
    next_emit_at = time.monotonic() + _HEALTH_PROGRESS_INTERVAL_S

    def on_poll(status_summary: str) -> None:
        nonlocal next_emit_at
        if time.monotonic() >= next_emit_at:
            on_progress(f"still waiting for Airflow to report healthy (latest: {status_summary})")
            next_emit_at = time.monotonic() + _HEALTH_PROGRESS_INTERVAL_S

    return on_poll


def start_runtime(
    config: RuntimeConfig,
    bundle_dir: Path | str,
    *,
    project_name: str | None = None,
    wait_timeout_s: float = 300.0,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[BundlePaths, AirflowHealth]:
    """Render the bundle, ``docker compose up -d``, and wait until healthy.

    The first start pulls the Airflow and Postgres images (minutes) and builds
    the user venv; later starts reuse both. ``on_progress`` (if given) receives
    plain-sentence narration of each phase so a first `up` never looks hung.
    """

    def emit_progress(message: str) -> None:
        if on_progress is not None:
            on_progress(message)

    emit_progress(f"rendering the runtime bundle at {Path(bundle_dir)}")
    paths = render_bundle(config, bundle_dir)
    emit_progress(
        "starting containers -- the first run downloads ~2 GB of images and builds "
        "the task venv (one-time; app.test() needs none of this)"
    )
    compose_up_detached(paths.compose_file, project_name=project_name)
    started_at = time.monotonic()
    client = client_for_bundle(paths)
    health = client.wait_until_healthy(
        timeout_s=wait_timeout_s,
        on_poll=_throttled_health_progress(on_progress) if on_progress is not None else None,
    )
    emit_progress("waiting for the ingest DAGs to register")
    # Component health says nothing about whether the generated DAGs actually
    # imported; only declare victory once the master AND its four stage
    # sub-DAGs are triggerable, so an immediate `hflow ingest` never 404s
    # and the master's first trigger task never fires at an unregistered
    # sub-DAG.
    remaining_s = max(30.0, wait_timeout_s - (time.monotonic() - started_at))
    _wait_until_dag_registered(client, bundle_dag_ids(paths.dag_id), timeout_s=remaining_s)
    return paths, health


def _wait_until_dag_registered(
    client: AirflowClient,
    dag_ids: Sequence[str],
    *,
    timeout_s: float,
    poll_interval_s: float = 3.0,
) -> None:
    """Poll every id within the one shared budget until all are registered."""
    deadline = time.monotonic() + timeout_s
    pending_dag_ids = list(dag_ids)
    last_error = "not yet polled"
    while time.monotonic() < deadline:
        still_pending: list[str] = []
        for dag_id in pending_dag_ids:
            try:
                client.dag(dag_id)
            except AirflowClientError as error:
                last_error = f"{dag_id}: {error}"
                still_pending.append(dag_id)
        pending_dag_ids = still_pending
        if not pending_dag_ids:
            return
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            break
        time.sleep(min(poll_interval_s, remaining_s))
    raise TimeoutError(
        f"Airflow is healthy but the ingest DAG(s) {pending_dag_ids!r} never registered "
        f"within {timeout_s:.0f}s (last: {last_error}) -- the DAG file likely "
        "failed to import; check `docker compose logs airflow-dag-processor`"
    )


def started_summary(paths: BundlePaths) -> str:
    """What a user needs after ``up``: UI URL, credentials, DAG id."""
    return "\n".join(
        [
            f"Airflow UI:  {paths.api_base_url}",
            f"credentials: {paths.admin_username} / {paths.admin_password}",
            f"ingest DAG:  {paths.dag_id}",
            f"bundle:      {paths.bundle_dir}",
            f"ingest with: hflow ingest <episode-uri> --bundle-dir {paths.bundle_dir}",
        ]
    )


# Plain-language diagnostics for the traps behind each unhealthy component
# (docs/ARCHITECTURE.md: the SDK explains, Airflow's UI observes).
_COMPONENT_HINTS: dict[str, str] = {
    "metadatabase": (
        "the metadata database is down, nothing can run -- check `docker compose logs postgres`"
    ),
    "scheduler": ("tasks will never start -- check `docker compose logs airflow-scheduler`"),
    "dag_processor": (
        "DAG files are not being parsed, so the ingest DAG never appears or "
        "updates -- check `docker compose logs airflow-dag-processor`"
    ),
    "triggerer": (
        "deferred tasks will never resume -- check `docker compose logs airflow-triggerer`"
    ),
}


def describe_runtime_status(paths: BundlePaths, *, project_name: str | None = None) -> str:
    """Health summary plus ``docker compose ps``, with plain-language hints."""
    lines = [f"bundle:  {paths.bundle_dir}", f"api:     {paths.api_base_url}"]
    try:
        health = client_for_bundle(paths).health()
    except AirflowClientError as error:
        lines.append(f"health:  unreachable ({error})")
        lines.append(
            "hint:    the api-server is not answering -- run `hflow up`, or check the "
            "service table below for exited containers"
        )
    else:
        overall = "healthy" if health.healthy else "UNHEALTHY"
        lines.append(f"health:  {overall} ({health.summary()})")
        for component_name, component_status in sorted(health.components.items()):
            if component_status != "healthy" and component_name in _COMPONENT_HINTS:
                lines.append(
                    f"hint:    {component_name} is {component_status or 'absent'}: "
                    f"{_COMPONENT_HINTS[component_name]}"
                )
    try:
        service_table = compose_ps(paths.compose_file, project_name=project_name)
    except ComposeError as error:
        service_table = str(error)
    lines.extend(["", service_table.rstrip()])
    return "\n".join(lines)
