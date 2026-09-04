"""Canonical episode format constants.

Every identifier in this module is written into MCAP files and must therefore
be neutral and format-versioned: it never embeds the project name, and
a breaking change to a record's contents bumps the
``/v1`` suffix rather than mutating the meaning of existing files.

The canonical episode convention (see docs/ARCHITECTURE.md, "The episode
container"):

- Camera streams are in-band H.264 in ``foxglove.CompressedVideo`` messages
  (protobuf-encoded): Annex B byte stream, SPS/PPS on every keyframe, no
  B-frames, one decodable access unit per message.
- Topics are assigned to named chunk groups; topics in different groups never
  share a chunk. Default grouping: camera schemas in ``cameras``, everything
  else in ``state``.
- Episode semantics and version stamps live in MCAP Metadata records named
  ``episode/v1`` and ``provenance/v1``; calibration/URDF files in Attachments.
"""

from enum import StrEnum

EPISODE_FORMAT_VERSION = "1"

# Episode semantics: task, operator, success, embodiment,
# robot_software_version, plus any user-supplied keys. All values are strings.
METADATA_RECORD_EPISODE = "episode/v1"

# What produced this file: schema_version, pipeline_version (content hash of
# the producing step configuration), ffmpeg_version, gop_preset, gop_seconds,
# source_uri. All values are strings.
METADATA_RECORD_PROVENANCE = "provenance/v1"

PROVENANCE_KEY_SCHEMA_VERSION = "schema_version"
PROVENANCE_KEY_PIPELINE_VERSION = "pipeline_version"
EPISODE_KEY_ROBOT_SOFTWARE_VERSION = "robot_software_version"

# Stamped as ``ffmpeg_version`` when the transform did no video work. The
# provenance record must stay honest -- ffmpeg was not an instrument for such
# a file -- and resolving a version anyway would force the pinned-build
# download for camera-less episodes.
FFMPEG_VERSION_NOT_USED = "not-used"

# Log times, publish times, and every resampling grid are nanoseconds since
# the epoch, so the conversion factor is shared rather than re-spelled.
NANOSECONDS_PER_SECOND = 1_000_000_000

# Version of the grid-resampling alignment policy (``hflow.resample``).
# Multi-rate alignment is where format converters silently diverge
# (docs/ARCHITECTURE.md, "Transform"), so the policy is explicit and
# versioned: bump this when the grid or selection semantics change.
# Annotated as ``str`` rather than inferred as a literal, for the same reason
# TRANSFORM_BEHAVIOR_VERSION is: the whole point is that it changes.
RESAMPLE_POLICY_VERSION: str = "2"

# ``provenance/v1`` keys written when derived channels are present:
# ``derived/<topic>`` holds each derived channel's author-declared version, and
# ``resample_policy_version`` names the alignment semantics they used.
PROVENANCE_KEY_RESAMPLE_POLICY_VERSION = "resample_policy_version"
PROVENANCE_KEY_DERIVED_PREFIX = "derived/"

# Which group each topic's channels were written into. Recorded because the
# assignment is now partly DATA-derived (see DEFAULT_BULK_GROUP): group names
# appear nowhere in the MCAP itself, so without this the resolved layout of a
# published episode could only be guessed at from chunk membership, and
# `hflow doctor`'s chunk-mixes-video-and-state finding could not be checked
# against what the transform intended.
PROVENANCE_KEY_TOPIC_GROUP_PREFIX = "group/"

# The chunk target each group was written at, recorded ONLY when the adaptive
# policy derived it. A configured target can be read back off the transform
# config; a derived one is a fact about this episode's own byte rates and
# exists nowhere else.
PROVENANCE_KEY_CHUNK_TARGET_PREFIX = "chunk-target/"

# Schema record name for derived channels (JSON messages on a grid). Neutral
# and format-versioned like every stored identifier in this module.
DERIVED_SCHEMA_NAME = "derived/v1"

# Version of the on-disk catalog layout (Parquet tables under the data root).
# Written to a marker file at the catalog root; bump on breaking layout change.
CATALOG_FORMAT_VERSION = "1"

DEFAULT_CAMERA_GROUP = "cameras"
DEFAULT_STATE_GROUP = "state"

# Bulk modalities (lidar and radar point clouds, occupancy grids) get their
# own group instead of sharing chunks with proprio-sized telemetry. Grouping
# is a READ-PATTERN decision, not a schema one: a training sample co-reads
# cameras and small state channels, and never the point clouds, so mixing
# them makes every state read drag bulk bytes along. docs/BENCHMARKS.md
# measured exactly that on real footage -- an `/imu` scan pulled 230 MB
# through the reader because `/imu` shared chunks with lidar, against 4 MB
# once the bulk channels moved out.
DEFAULT_BULK_GROUP = "bulk"

# Mean payload bytes at or below which a non-camera channel counts as
# proprio-like telemetry that samples co-read. Above it, the channel is bulk.
# The threshold the benchmark's read-pattern mode used to produce the numbers
# in docs/BENCHMARKS.md.
BULK_MESSAGE_BYTES = 16_384

# Target uncompressed chunk size per group, in bytes. Small enough that a
# few-second time slice of a group is one or two fetches; measured defaults
# are a benchmark-report deliverable.
DEFAULT_CHUNK_SIZE_BYTES = 800_000

# Schema names that identify a topic as a camera stream for default group
# assignment and for ``Episode.cameras``.
CAMERA_SCHEMA_NAMES = frozenset(
    {
        "foxglove.CompressedVideo",
        "foxglove.CompressedImage",
        "foxglove.RawImage",
        "foxglove_msgs/msg/CompressedVideo",
        "sensor_msgs/msg/CompressedImage",
        "sensor_msgs/msg/Image",
    }
)

CANONICAL_VIDEO_SCHEMA_NAME = "foxglove.CompressedVideo"

# Already-encoded video schemas whose payload contract the built-in transform
# and doctor both understand. The ROS 2 spelling remains a source schema; the
# canonical writer emits the protobuf spelling above.
PASSTHROUGH_VIDEO_SCHEMA_NAMES = frozenset(
    {CANONICAL_VIDEO_SCHEMA_NAME, "foxglove_msgs/msg/CompressedVideo"}
)


class GopPreset(StrEnum):
    """GOP-length preset keyed to how the data will be read.

    GOP length is a training hyperparameter: VLA-style training samples short
    sparse windows (pays a keyframe
    seek per sample, wants short GOPs); world-model training reads long
    contiguous sequences (amortizes keyframes, wants long GOPs).
    """

    VLA = "vla"
    WORLD_MODEL = "world_model"


# Seconds between keyframes per preset. Configurable at the writer; these
# defaults are our engineering judgment pending the benchmark report.
GOP_SECONDS: dict[GopPreset, float] = {
    GopPreset.VLA: 1.0,
    GopPreset.WORLD_MODEL: 6.0,
}

# How much time one read of this workload covers. The same fact as GOP_SECONDS
# above -- the preset IS the read pattern -- applied to chunk sizing instead of
# keyframe spacing, and kept as its own mapping because the two answer
# different questions and there is no reason they must stay equal.
#
# VLA's 1.0 s is the benchmark's own sample window (benchmarks/read_benchmark.py)
# and the source of every published number. WORLD_MODEL's 6.0 s is the sequence
# length that preset already asserts; no 6 s window read has been measured, so
# treat that one as the preset's claim rather than as evidence.
READ_WINDOW_SECONDS: dict[GopPreset, float] = {
    GopPreset.VLA: 1.0,
    GopPreset.WORLD_MODEL: 6.0,
}

# Bounds on a derived chunk target. Both endpoints are chunk sizes with
# published measurements in docs/BENCHMARKS.md, so interpolating between them
# is supported and extrapolating past them is not.
#
# The floor being DEFAULT_CHUNK_SIZE_BYTES is load-bearing: every group whose
# byte rate falls below the floor keeps today's exact layout, which covers all
# state groups, every synthetic fixture, and single-camera episodes.
MINIMUM_DERIVED_CHUNK_SIZE_BYTES = DEFAULT_CHUNK_SIZE_BYTES
MAXIMUM_DERIVED_CHUNK_SIZE_BYTES = 8_000_000


def derived_chunk_size_bytes(group_bytes_per_second: float, read_window_seconds: float) -> int:
    """The chunk target for a group written at ``group_bytes_per_second``.

    A read covering ``W`` seconds of a group written at rate ``R`` with chunk
    target ``C`` costs about ``1 + R*W/C`` fetches (the chunks its payload
    fills, plus the partial chunk it starts inside) and about ``C + R*W``
    bytes. Neither is minimized alone: fetches fall forever as ``C`` grows,
    bytes fall forever as ``C`` shrinks. Writing ``x = C/(R*W)``, their product
    is ``R*W*(2 + x + 1/x)``, which is minimized exactly at ``x = 1``. So the
    balance point is ``C = R*W``: two fetches per group per window, for twice
    the bytes the window actually needs.

    That model reproduces both measured rows in docs/BENCHMARKS.md from one
    fit (7.8 predicted vs 9.2 measured at 800 KB, 2.7 vs 2.7 at 8 MB), which
    is licence to interpolate between them and no further -- hence the clamp.

    The target is in UNCOMPRESSED bytes, because that is what the writer
    accumulates before it compresses. The quantity a reader ultimately pays is
    compressed, so a group that compresses well fetches proportionally less
    than this predicts; the ratio is the same for every candidate target, so
    it moves the absolute numbers and not the choice between them.
    """
    balanced_target = group_bytes_per_second * read_window_seconds
    return int(
        min(
            MAXIMUM_DERIVED_CHUNK_SIZE_BYTES,
            max(MINIMUM_DERIVED_CHUNK_SIZE_BYTES, balanced_target),
        )
    )
