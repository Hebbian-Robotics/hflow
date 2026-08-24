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
  share a chunk (topic-group chunking, described in Dyna's article). Default
  grouping: camera schemas in ``cameras``, everything else in ``state``.
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
# ``derived/<topic>`` holds each derived channel's content-hash version, and
# ``resample_policy_version`` names the alignment semantics they used.
PROVENANCE_KEY_RESAMPLE_POLICY_VERSION = "resample_policy_version"
PROVENANCE_KEY_DERIVED_PREFIX = "derived/"

# Schema record name for derived channels (JSON messages on a grid). Neutral
# and format-versioned like every stored identifier in this module.
DERIVED_SCHEMA_NAME = "derived/v1"

# Version of the on-disk catalog layout (Parquet tables under the data root).
# Written to a marker file at the catalog root; bump on breaking layout change.
CATALOG_FORMAT_VERSION = "1"

DEFAULT_CAMERA_GROUP = "cameras"
DEFAULT_STATE_GROUP = "state"

# Target uncompressed chunk size per group, in bytes. Small enough that a
# few-second time slice of a group is one or two fetches; measured defaults
# are a benchmark-report deliverable (Dyna's article discloses no numbers).
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


class GopPreset(StrEnum):
    """GOP-length preset keyed to how the data will be read.

    GOP length is a training hyperparameter (Dyna's article treats it as
    one): VLA-style training samples short sparse windows (pays a keyframe
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
