"""The Docker Compose runtime: provision and talk to a local Airflow.

``hflow up`` renders a bundle (see ``_bundle``) and starts it with plain
``docker compose``; observability is Airflow's own UI on localhost. The SDK
never imports Airflow -- it renders files and speaks the REST API.
"""

from hflow.runtime._bundle import (
    DEFAULT_AIRFLOW_IMAGE,
    BundlePaths,
    RuntimeConfig,
    bundle_dag_ids,
    find_bundle_directory,
    infer_hflow_source,
    load_bundle,
    render_bundle,
    sub_dag_id_for_stage,
)
from hflow.runtime._client import (
    AirflowAuth,
    AirflowClient,
    AirflowClientError,
    AirflowHealth,
    BearerToken,
    PasswordCredentials,
)
from hflow.runtime._compose import (
    ComposeError,
    compose_down,
    compose_logs,
    compose_ps,
    compose_up_detached,
)
from hflow.runtime._deploy import (
    DeployConfig,
    DeployPaths,
    render_deploy_bundle,
    validate_data_root_uri,
)
from hflow.runtime._endpoint import (
    RemoteRuntimeEndpoint,
    client_for_endpoint,
    describe_remote_status,
    resolve_remote_endpoint,
)
from hflow.runtime._lifecycle import (
    client_for_bundle,
    describe_runtime_status,
    start_runtime,
    started_summary,
)
from hflow.runtime._topology import (
    DagTaskNode,
    DagTopology,
    IngestTopology,
    StageTopology,
    ingest_dag_topology,
)
from hflow.uri import DataRootRelativeUri, parse_data_root_relative_uri

__all__ = [
    "DEFAULT_AIRFLOW_IMAGE",
    "AirflowAuth",
    "AirflowClient",
    "AirflowClientError",
    "AirflowHealth",
    "BearerToken",
    "BundlePaths",
    "ComposeError",
    "DagTaskNode",
    "DagTopology",
    "DataRootRelativeUri",
    "DeployConfig",
    "DeployPaths",
    "IngestTopology",
    "PasswordCredentials",
    "RemoteRuntimeEndpoint",
    "RuntimeConfig",
    "StageTopology",
    "bundle_dag_ids",
    "client_for_bundle",
    "client_for_endpoint",
    "compose_down",
    "compose_logs",
    "compose_ps",
    "compose_up_detached",
    "describe_remote_status",
    "describe_runtime_status",
    "find_bundle_directory",
    "infer_hflow_source",
    "ingest_dag_topology",
    "load_bundle",
    "parse_data_root_relative_uri",
    "render_bundle",
    "render_deploy_bundle",
    "resolve_remote_endpoint",
    "start_runtime",
    "started_summary",
    "sub_dag_id_for_stage",
    "validate_data_root_uri",
]
