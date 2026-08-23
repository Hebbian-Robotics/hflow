"""hflow: an open-source robotics data pipeline.

Ingest, quality-check, enrich, and curate robot episode data. See README.md
for orientation and docs/ARCHITECTURE.md for the design and its references.
"""

from importlib.metadata import PackageNotFoundError, version

from hflow import checks, ffmpeg, providers, testing
from hflow.app import (
    App,
    CheckRunReport,
    CheckStatus,
    EnrichmentRunReport,
    TestReport,
    import_pipeline_application,
)
from hflow.batching import PlannedBatch, plan_batches, plan_batches_from_files
from hflow.catalog import AppendResult, Catalog, CheckRunRow
from hflow.curation import (
    CheckCoverage,
    CurationReport,
    StaleEpisode,
    curate,
    open_catalog_connection,
    stale_episodes,
)
from hflow.doctor import DiagnosticLevel, DoctorReport, Finding, diagnose
from hflow.episode import ChannelData, Episode, ExtractedFrame
from hflow.format import GopPreset
from hflow.manifest import (
    DerivedChannelManifest,
    PipelineManifest,
    StepKind,
    StepManifest,
)
from hflow.mcap_writer import CanonicalMcapWriter
from hflow.reader import (
    EpisodeReader,
    MessageBatch,
    PythonMcapEpisodeReader,
    TopicInfo,
    open_reader,
)
from hflow.resample import DerivedSeries, ResamplePolicy, to_grid
from hflow.review import (
    REVIEW_DATASET_FORMAT_NAME,
    REVIEW_DATASET_FORMAT_VERSION,
    ReviewDatasetReport,
    ReviewMediaMode,
    export_review_dataset,
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
    "REVIEW_DATASET_FORMAT_NAME",
    "REVIEW_DATASET_FORMAT_VERSION",
    "RUN_PROFILES",
    "Aggregation",
    "App",
    "AppendResult",
    "BucketStorageRoot",
    "CanonicalMcapWriter",
    "Catalog",
    "ChannelData",
    "CheckCoverage",
    "CheckResult",
    "CheckRunReport",
    "CheckRunRow",
    "CheckStatus",
    "Comparison",
    "CurationReport",
    "DerivedChannel",
    "DerivedChannelManifest",
    "DerivedSeries",
    "DiagnosticLevel",
    "DoctorReport",
    "EnrichmentResult",
    "EnrichmentRunReport",
    "Episode",
    "EpisodeReader",
    "EpisodeStamps",
    "ExtractedFrame",
    "Finding",
    "Gate",
    "GateAbstained",
    "GateDecided",
    "GopPreset",
    "IngestMode",
    "Interval",
    "LocalStorageRoot",
    "MeasurementValue",
    "MessageBatch",
    "PipelineManifest",
    "PlannedBatch",
    "PythonMcapEpisodeReader",
    "RegisteredCheck",
    "RegisteredEnrichment",
    "ResamplePolicy",
    "ReviewDatasetReport",
    "ReviewMediaMode",
    "Stage",
    "StaleEpisode",
    "StepKind",
    "StepManifest",
    "StorageRoot",
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
    "export_review_dataset",
    "fetch_uri",
    "ffmpeg",
    "import_pipeline_application",
    "is_bucket_url",
    "open_catalog_connection",
    "open_reader",
    "parse_storage_root",
    "plan_batches",
    "plan_batches_from_files",
    "providers",
    "stages_for_profile",
    "stale_episodes",
    "testing",
    "to_grid",
    "write_canonical_episode",
]
