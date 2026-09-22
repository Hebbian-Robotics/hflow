"""hflow: an open-source robotics data pipeline.

Ingest, quality-check, enrich, and curate robot episode data. See README.md
for orientation and docs/ARCHITECTURE.md for the design and its references.
"""

import logging
from importlib import import_module
from typing import TYPE_CHECKING

from hflow._version import __version__

logging.getLogger(__name__).addHandler(logging.NullHandler())

if TYPE_CHECKING:
    from hflow import build_ai_vlm_checks, checks, ffmpeg, providers, testing
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
        ProcessManyProgress,
        ProcessManyReport,
        ProcessReport,
        PublishFailed,
        SkippedByQuarantine,
        StepNotRun,
        SupersededByPipeline,
        TestManyProgress,
        TestManyReport,
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
    from hflow.fingerprints import (
        ContractFingerprint,
        fingerprint_contract,
        step_version_from_contract,
    )
    from hflow.format import GopPreset
    from hflow.importers import (
        VideoImportConfig,
        import_lerobot_dataset,
        import_video_episode,
        verify_lerobot_import,
    )
    from hflow.manifest import (
        DerivedChannelManifest,
        PipelineManifest,
        StepKind,
        StepManifest,
    )
    from hflow.reader import (
        EpisodeReader,
        EpisodeTimeBounds,
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
        verify_dataset_snapshot,
    )
    from hflow.source_sampling import (
        SOURCE_FRAME_SAMPLING_VERSION,
        KeyframeFallbackReason,
        SampledSourceFrame,
        SourceFrameSamples,
        SourceFrameSampling,
        SourceSamplingError,
        SourceSamplingMode,
        sample_source_frames,
    )
    from hflow.source_windows import SourceWindow, plan_source_windows
    from hflow.statistics import (
        WeightedDistribution,
        WeightedHistogramBin,
        WeightedPercentile,
        WeightedValue,
        summarize_weighted_distribution,
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
        StepVersion,
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
    from hflow.verification import (
        VerificationFinding,
        VerificationReason,
        VerificationReport,
        VerificationStatus,
    )
    from hflow.workspace import Workspace, WorkspaceIdentity

_ATTRIBUTE_NAMES_BY_MODULE: dict[str, tuple[str, ...]] = {
    "hflow.app": (
        "App",
        "CheckOutcome",
        "CheckRunReport",
        "CheckStatus",
        "EnrichmentOutcome",
        "EnrichmentRunReport",
        "Errored",
        "Measured",
        "NotRun",
        "ProcessManyProgress",
        "ProcessManyReport",
        "ProcessReport",
        "PublishFailed",
        "SkippedByQuarantine",
        "StepNotRun",
        "SupersededByPipeline",
        "TestManyProgress",
        "TestManyReport",
        "TestReport",
        "import_pipeline_application",
    ),
    "hflow.batching": (
        "PlannedBatch",
        "plan_batches",
        "plan_batches_from_files",
    ),
    "hflow.catalog": (
        "AppendResult",
        "Catalog",
        "CheckRunRow",
    ),
    "hflow.catalog_ui": (
        "DEFAULT_CATALOG_UI_PORT",
        "CatalogUiSettings",
        "CatalogUiStartupError",
        "serve_catalog_ui",
    ),
    "hflow.curation": (
        "CheckCoverage",
        "CurationReport",
        "StaleEpisode",
        "curate",
        "open_catalog_connection",
        "stale_episodes",
    ),
    "hflow.doctor": (
        "DiagnosticLevel",
        "DoctorReport",
        "Finding",
        "diagnose",
    ),
    "hflow.episode": (
        "ChannelData",
        "DecodedMessageBatch",
        "Episode",
        "ExtractedFrame",
    ),
    "hflow.fingerprints": (
        "ContractFingerprint",
        "fingerprint_contract",
        "step_version_from_contract",
    ),
    "hflow.format": ("GopPreset",),
    "hflow.importers": (
        "VideoImportConfig",
        "import_lerobot_dataset",
        "import_video_episode",
        "verify_lerobot_import",
    ),
    "hflow.manifest": (
        "DerivedChannelManifest",
        "PipelineManifest",
        "StepKind",
        "StepManifest",
    ),
    "hflow.reader": (
        "EpisodeReader",
        "EpisodeTimeBounds",
        "MessageBatch",
        "PythonMcapEpisodeReader",
        "TopicInfo",
        "open_reader",
    ),
    "hflow.resample": (
        "DerivedSeries",
        "ResamplePolicy",
        "to_grid",
    ),
    "hflow.snapshot": (
        "DATASET_SNAPSHOT_FORMAT_NAME",
        "DATASET_SNAPSHOT_FORMAT_VERSION",
        "DatasetSnapshotReport",
        "RetainedDatasetSnapshotBackup",
        "SnapshotMediaMode",
        "export_dataset_snapshot",
        "verify_dataset_snapshot",
    ),
    "hflow.source_sampling": (
        "SOURCE_FRAME_SAMPLING_VERSION",
        "KeyframeFallbackReason",
        "SampledSourceFrame",
        "SourceFrameSamples",
        "SourceFrameSampling",
        "SourceSamplingError",
        "SourceSamplingMode",
        "sample_source_frames",
    ),
    "hflow.source_windows": (
        "SourceWindow",
        "plan_source_windows",
    ),
    "hflow.statistics": (
        "WeightedDistribution",
        "WeightedHistogramBin",
        "WeightedPercentile",
        "WeightedValue",
        "summarize_weighted_distribution",
    ),
    "hflow.steps": (
        "RUN_PROFILES",
        "Aggregation",
        "CheckResult",
        "Comparison",
        "DerivedChannel",
        "EnrichmentResult",
        "Gate",
        "GateAbstained",
        "GateDecided",
        "IngestMode",
        "Interval",
        "MeasurementValue",
        "Observation",
        "RegisteredCheck",
        "RegisteredEnrichment",
        "Stage",
        "StepVersion",
        "Threshold",
        "evaluate_gate",
        "stages_for_profile",
    ),
    "hflow.storage": (
        "BucketStorageRoot",
        "LocalStorageRoot",
        "StorageRoot",
        "fetch_uri",
        "is_bucket_url",
        "parse_storage_root",
    ),
    "hflow.transform": (
        "EpisodeStamps",
        "TransformConfig",
        "write_canonical_episode",
    ),
    "hflow.verification": (
        "VerificationFinding",
        "VerificationReason",
        "VerificationReport",
        "VerificationStatus",
    ),
    "hflow.workspace": (
        "Workspace",
        "WorkspaceIdentity",
    ),
}
_MODULE_EXPORTS: dict[str, str] = {
    module_name.removeprefix("hflow."): module_name for module_name in _ATTRIBUTE_NAMES_BY_MODULE
}
_MODULE_EXPORTS.update(
    {
        "build_ai_vlm_checks": "hflow.build_ai_vlm_checks",
        "checks": "hflow.checks",
        "ffmpeg": "hflow.ffmpeg",
        "providers": "hflow.providers",
        "testing": "hflow.testing",
    }
)
_EXPORT_MODULE_BY_NAME = {
    name: module_name
    for module_name, attribute_names in _ATTRIBUTE_NAMES_BY_MODULE.items()
    for name in attribute_names
}


def __getattr__(name: str) -> object:
    """Resolve public exports only when their functionality is requested."""
    if name in _MODULE_EXPORTS:
        exported_value = import_module(_MODULE_EXPORTS[name])
    elif name in _EXPORT_MODULE_BY_NAME:
        exported_value = getattr(import_module(_EXPORT_MODULE_BY_NAME[name]), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = exported_value
    return exported_value


def __dir__() -> list[str]:
    """Keep public exports discoverable without importing their implementations."""
    return sorted(set(globals()) | set(__all__) | set(_MODULE_EXPORTS))


__all__ = [
    "DATASET_SNAPSHOT_FORMAT_NAME",
    "DATASET_SNAPSHOT_FORMAT_VERSION",
    "DEFAULT_CATALOG_UI_PORT",
    "RUN_PROFILES",
    "SOURCE_FRAME_SAMPLING_VERSION",
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
    "ContractFingerprint",
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
    "EpisodeTimeBounds",
    "Errored",
    "ExtractedFrame",
    "Finding",
    "Gate",
    "GateAbstained",
    "GateDecided",
    "GopPreset",
    "IngestMode",
    "Interval",
    "KeyframeFallbackReason",
    "LocalStorageRoot",
    "Measured",
    "MeasurementValue",
    "MessageBatch",
    "NotRun",
    "Observation",
    "PipelineManifest",
    "PlannedBatch",
    "ProcessManyProgress",
    "ProcessManyReport",
    "ProcessReport",
    "PublishFailed",
    "PythonMcapEpisodeReader",
    "RegisteredCheck",
    "RegisteredEnrichment",
    "ResamplePolicy",
    "RetainedDatasetSnapshotBackup",
    "SampledSourceFrame",
    "SkippedByQuarantine",
    "SnapshotMediaMode",
    "SourceFrameSamples",
    "SourceFrameSampling",
    "SourceSamplingError",
    "SourceSamplingMode",
    "SourceWindow",
    "Stage",
    "StaleEpisode",
    "StepKind",
    "StepManifest",
    "StepNotRun",
    "StepVersion",
    "StorageRoot",
    "SupersededByPipeline",
    "TestManyProgress",
    "TestManyReport",
    "TestReport",
    "Threshold",
    "TopicInfo",
    "TransformConfig",
    "VerificationFinding",
    "VerificationReason",
    "VerificationReport",
    "VerificationStatus",
    "VideoImportConfig",
    "WeightedDistribution",
    "WeightedHistogramBin",
    "WeightedPercentile",
    "WeightedValue",
    "Workspace",
    "WorkspaceIdentity",
    "__version__",
    "build_ai_vlm_checks",
    "checks",
    "curate",
    "diagnose",
    "evaluate_gate",
    "export_dataset_snapshot",
    "fetch_uri",
    "ffmpeg",
    "fingerprint_contract",
    "import_lerobot_dataset",
    "import_pipeline_application",
    "import_video_episode",
    "is_bucket_url",
    "open_catalog_connection",
    "open_reader",
    "parse_storage_root",
    "plan_batches",
    "plan_batches_from_files",
    "plan_source_windows",
    "providers",
    "sample_source_frames",
    "serve_catalog_ui",
    "stages_for_profile",
    "stale_episodes",
    "step_version_from_contract",
    "summarize_weighted_distribution",
    "testing",
    "to_grid",
    "verify_dataset_snapshot",
    "verify_lerobot_import",
    "write_canonical_episode",
]
