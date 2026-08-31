"""hflow: an open-source robotics data pipeline.

Ingest, quality-check, enrich, and curate robot episode data. See README.md
for orientation and docs/ARCHITECTURE.md for the design and its references.
"""

from importlib.metadata import PackageNotFoundError, version

from hflow import checks, ffmpeg, providers, testing
from hflow.app import (
    App,
    CheckOutcome,
    CheckRunReport,
    CheckStatus,
    EnrichmentOutcome,
    EnrichmentRunReport,
    Errored,
    Measured,
    NotRun,
    PublishFailed,
    SkippedByQuarantine,
    StepNotRun,
    SupersededByPipeline,
    TestReport,
    import_pipeline_application,
)
from hflow.batching import PlannedBatch, plan_batches, plan_batches_from_files
from hflow.catalog import AppendResult, Catalog, CheckRunRow
from hflow.catalog_ui import (
    DEFAULT_CATALOG_UI_PORT,
    CatalogUiSettings,
    CatalogUiStartupError,
    serve_catalog_ui,
)
from hflow.curation import (
    CheckCoverage,
    CurationReport,
    StaleEpisode,
    curate,
    open_catalog_connection,
    stale_episodes,
)
from hflow.doctor import DiagnosticLevel, DoctorReport, Finding, diagnose
from hflow.episode import ChannelData, DecodedMessageBatch, Episode, ExtractedFrame
from hflow.format import GopPreset
from hflow.importers import import_lerobot_dataset
from hflow.manifest import (
    DerivedChannelManifest,
    PipelineManifest,
    StepKind,
    StepManifest,
)
from hflow.reader import (
    EpisodeReader,
    MessageBatch,
    PythonMcapEpisodeReader,
    TopicInfo,
    open_reader,
)
from hflow.resample import DerivedSeries, ResamplePolicy, to_grid
from hflow.snapshot import (
    DATASET_SNAPSHOT_FORMAT_NAME,
    DATASET_SNAPSHOT_FORMAT_VERSION,
    DatasetSnapshotReport,
    RetainedDatasetSnapshotBackup,
    SnapshotMediaMode,
    export_dataset_snapshot,
)
from hflow.steps import (
    RUN_PROFILES,
    Aggregation,
    CheckResult,
    Comparison,
    DerivedChannel,
    EnrichmentResult,
    Gate,
    GateAbstained,
    GateDecided,
    IngestMode,
    Interval,
    MeasurementValue,
    Observation,
    RegisteredCheck,
    RegisteredEnrichment,
    Stage,
    Threshold,
    evaluate_gate,
    stages_for_profile,
)
from hflow.storage import (
    BucketStorageRoot,
    LocalStorageRoot,
    StorageRoot,
    fetch_uri,
    is_bucket_url,
    parse_storage_root,
)
from hflow.transform import EpisodeStamps, TransformConfig, write_canonical_episode
from hflow.workspace import Workspace, WorkspaceIdentity

try:
    __version__ = version("hflow")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0"

__all__ = [
    "DATASET_SNAPSHOT_FORMAT_NAME",
    "DATASET_SNAPSHOT_FORMAT_VERSION",
    "DEFAULT_CATALOG_UI_PORT",
    "RUN_PROFILES",
    "Aggregation",
    "App",
    "AppendResult",
    "BucketStorageRoot",
    "Catalog",
    "CatalogUiSettings",
    "CatalogUiStartupError",
    "ChannelData",
    "CheckCoverage",
    "CheckOutcome",
    "CheckResult",
    "CheckRunReport",
    "CheckRunRow",
    "CheckStatus",
    "Comparison",
    "CurationReport",
    "DatasetSnapshotReport",
    "DecodedMessageBatch",
    "DerivedChannel",
    "DerivedChannelManifest",
    "DerivedSeries",
    "DiagnosticLevel",
    "DoctorReport",
    "EnrichmentOutcome",
    "EnrichmentResult",
    "EnrichmentRunReport",
    "Episode",
    "EpisodeReader",
    "EpisodeStamps",
    "Errored",
    "ExtractedFrame",
    "Finding",
    "Gate",
    "GateAbstained",
    "GateDecided",
    "GopPreset",
    "IngestMode",
    "Interval",
    "LocalStorageRoot",
    "Measured",
    "MeasurementValue",
    "MessageBatch",
    "NotRun",
    "Observation",
    "PipelineManifest",
    "PlannedBatch",
    "PublishFailed",
    "PythonMcapEpisodeReader",
    "RegisteredCheck",
    "RegisteredEnrichment",
    "ResamplePolicy",
    "RetainedDatasetSnapshotBackup",
    "SkippedByQuarantine",
    "SnapshotMediaMode",
    "Stage",
    "StaleEpisode",
    "StepKind",
    "StepManifest",
    "StepNotRun",
    "StorageRoot",
    "SupersededByPipeline",
    "TestReport",
    "Threshold",
    "TopicInfo",
    "TransformConfig",
    "Workspace",
    "WorkspaceIdentity",
    "__version__",
    "checks",
    "curate",
    "diagnose",
    "evaluate_gate",
    "export_dataset_snapshot",
    "fetch_uri",
    "ffmpeg",
    "import_lerobot_dataset",
    "import_pipeline_application",
    "is_bucket_url",
    "open_catalog_connection",
    "open_reader",
    "parse_storage_root",
    "plan_batches",
    "plan_batches_from_files",
    "providers",
    "serve_catalog_ui",
    "stages_for_profile",
    "stale_episodes",
    "testing",
    "to_grid",
    "write_canonical_episode",
]
