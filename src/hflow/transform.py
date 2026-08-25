"""The built-in transform: any input MCAP in, one canonical episode out.

v1 behavior:

- Camera image channels (``sensor_msgs/msg/CompressedImage`` /
  ``foxglove.CompressedImage``, JPEG or PNG payloads) are transcoded to
  in-band H.264 ``foxglove.CompressedVideo`` (protobuf) on the same topic,
  GOP length from the preset and the stream's measured frame rate.
- Channels already carrying ``foxglove.CompressedVideo`` pass through
  byte-for-byte, but only after every message is validated against the
  canonical video constraints (``format="h264"``, one AUD-delimited access
  unit per message, SPS+PPS on keyframes, each channel starts on a keyframe) --
  the provenance stamp asserts conformance, so nonconforming video must fail
  loudly, never pass silently.
  Raw ``sensor_msgs/msg/Image``/``foxglove.RawImage`` is not supported in v1
  (explicit error).
- Every other channel passes through byte-for-byte with its original schema.
  The transform is keyed by CHANNEL, not topic: a source with several
  channels sharing a topic produces the same number of output channels on
  that topic, each with its original schema intact. (The old v1
  one-channel-per-topic requirement existed only because the reader seam was
  topic-keyed.) Per-message ``sequence`` numbers and per-channel metadata
  maps are not carried through the batch reader seam and are dropped; both
  are rare and advisory.
- Topic-group chunking: camera-schema topics to the ``cameras`` group, bulk
  non-camera channels (mean payload over ``BULK_MESSAGE_BYTES`` -- point
  clouds, occupancy grids) to ``bulk``, everything else to ``state``, with
  per-topic overrides in the config that beat all of it (an override applies
  to every channel of that topic). The resolved topic-to-group map is written
  into ``provenance/v1``, because the assignment is partly data-derived and
  group names appear nowhere in the MCAP itself.
- Derived signals (``derived=``): each :class:`~hflow.resample.DerivedSeries`
  becomes a NEW channel in the state group -- message encoding ``json``, one
  message per grid point, schema ``derived/v1`` (jsonschema). Each derived
  channel's version and the resample policy version are stamped into
  ``provenance/v1``, and the derived topics+versions fold into
  ``pipeline_version`` so a changed derivation changes the content hash.
- ``ffmpeg_version`` is stamped only when video work actually happened (a
  transcode ran); otherwise the literal ``FFMPEG_VERSION_NOT_USED`` is
  written, so camera-less episodes never trigger the pinned-ffmpeg download.
- Source Metadata records and Attachments are copied through -- except a
  stale ``provenance/v1``, which is replaced by this transform's own stamps
  (schema_version, pipeline_version, ffmpeg_version, gop_preset, gop_seconds,
  source_uri).
- Messages are written in global log-time order; the writer distributes them
  to per-group chunk streams.
"""

import hashlib
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from hflow import video as video_module
from hflow.behavior import TRANSFORM_BEHAVIOR_VERSION
from hflow.ffmpeg import ffmpeg_version
from hflow.format import (
    BULK_MESSAGE_BYTES,
    CAMERA_SCHEMA_NAMES,
    CANONICAL_VIDEO_SCHEMA_NAME,
    DEFAULT_BULK_GROUP,
    DEFAULT_CAMERA_GROUP,
    DEFAULT_CHUNK_SIZE_BYTES,
    DEFAULT_STATE_GROUP,
    DERIVED_SCHEMA_NAME,
    EPISODE_FORMAT_VERSION,
    EPISODE_KEY_ROBOT_SOFTWARE_VERSION,
    FFMPEG_VERSION_NOT_USED,
    GOP_SECONDS,
    METADATA_RECORD_EPISODE,
    METADATA_RECORD_PROVENANCE,
    MINIMUM_DERIVED_CHUNK_SIZE_BYTES,
    PROVENANCE_KEY_CHUNK_TARGET_PREFIX,
    PROVENANCE_KEY_DERIVED_PREFIX,
    PROVENANCE_KEY_PIPELINE_VERSION,
    PROVENANCE_KEY_RESAMPLE_POLICY_VERSION,
    PROVENANCE_KEY_SCHEMA_VERSION,
    PROVENANCE_KEY_TOPIC_GROUP_PREFIX,
    READ_WINDOW_SECONDS,
    RESAMPLE_POLICY_VERSION,
    GopPreset,
    derived_chunk_size_bytes,
)
from hflow.mcap_writer import CanonicalMcapWriter
from hflow.reader import TopicInfo, open_reader
from hflow.resample import DerivedSeries

logger = logging.getLogger(__name__)

# Image-carrying schemas the v1 transform can transcode to in-band H.264.
_TRANSCODABLE_IMAGE_SCHEMAS = frozenset(
    {"sensor_msgs/msg/CompressedImage", "foxglove.CompressedImage"}
)
_RAW_IMAGE_SCHEMAS = frozenset({"sensor_msgs/msg/Image", "foxglove.RawImage"})
# Already-encoded video schemas that pass through (validated, not transcoded).
_PASSTHROUGH_VIDEO_SCHEMAS = frozenset(
    {CANONICAL_VIDEO_SCHEMA_NAME, "foxglove_msgs/msg/CompressedVideo"}
)


@dataclass(frozen=True)
class TransformConfig:
    """Configuration for :func:`write_canonical_episode`. This is the
    "step configuration" whose content hash becomes ``pipeline_version``."""

    gop_preset: GopPreset = GopPreset.VLA
    gop_seconds: float | None = None  # overrides the preset's default when set
    crf: int = 23
    # One target for every group, or ``None`` to derive one PER GROUP from the
    # rate that group is written at and the preset's read window (see
    # ``hflow.format.derived_chunk_size_bytes``). Deriving is opt-in rather
    # than the default because the balance point it computes has not been
    # measured on real footage: docs/BENCHMARKS.md has rows at 800 KB and
    # 8 MB, and the derived target for that recording falls between them.
    chunk_size_bytes: int | None = DEFAULT_CHUNK_SIZE_BYTES
    compression: Literal["zstd", "none"] = "zstd"
    # topic -> group name, beating the defaults; topics not listed go to
    # ``cameras`` by schema, ``bulk`` by mean message size, else ``state``.
    topic_groups: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeStamps:
    """The version stamps written into ``provenance/v1``."""

    schema_version: str
    pipeline_version: str
    ffmpeg_version: str
    robot_software_version: str | None


def stamps_from_provenance(metadata: dict[str, str]) -> EpisodeStamps:
    """Reconstruct :class:`EpisodeStamps` from a canonical episode's metadata.

    ``metadata`` is the merged flat dict ``Episode.metadata`` exposes (the
    ``episode/v1`` record merged with ``provenance/v1``), so an opened
    canonical Episode is enough. ``robot_software_version`` lives in the
    episode record and is optional; the provenance keys are not -- a file
    missing them was never written by the canonical transform.
    """
    try:
        return EpisodeStamps(
            schema_version=metadata[PROVENANCE_KEY_SCHEMA_VERSION],
            pipeline_version=metadata[PROVENANCE_KEY_PIPELINE_VERSION],
            ffmpeg_version=metadata["ffmpeg_version"],
            robot_software_version=metadata.get(EPISODE_KEY_ROBOT_SOFTWARE_VERSION),
        )
    except KeyError as error:
        raise ValueError(
            f"episode metadata is missing provenance key {error.args[0]!r}: this is "
            "not a canonical episode (no provenance/v1 record from the transform)"
        ) from error


def compute_pipeline_version(
    config: TransformConfig,
    derived_versions: Mapping[str, str] | None = None,
) -> str:
    """Content hash of the transform configuration (12 hex chars).

    ``derived_versions`` (derived topic -> derived-channel version) folds the
    derived signals into the hash: a passed-in mapping rather than App state,
    so the transform stays a library function.

    The engine's contribution is :data:`hflow.behavior.TRANSFORM_BEHAVIOR_VERSION`,
    NOT the release number: this hash is stamped into ``provenance/v1``, which
    lives inside the bytes ``content_episode_id`` hashes, so folding in
    ``hflow.__version__`` gave every release a new ``pipeline_version`` AND a
    new ``episode_id`` for byte-identical inputs -- breaking dedupe and making
    ``hflow stale`` list the whole corpus after a change that processed
    nothing differently.
    """
    payload = json.dumps(
        {
            "format": EPISODE_FORMAT_VERSION,
            "transform_behavior": TRANSFORM_BEHAVIOR_VERSION,
            "gop_preset": str(config.gop_preset),
            "gop_seconds": config.gop_seconds,
            "crf": config.crf,
            "chunk_size_bytes": config.chunk_size_bytes,
            "compression": config.compression,
            "topic_groups": dict(sorted(config.topic_groups.items())),
            "derived": dict(sorted(derived_versions.items())) if derived_versions else {},
            # Only when the episode HAS derived channels: the policy decides
            # their samples, so a bump must make them stale -- but folding it
            # unconditionally would re-version corpora that never resampled
            # anything. A step's own version cannot cover this, because the
            # taught idiom reaches ``hflow.resample.to_grid`` through the
            # module and a module contributes only its name to a step hash.
            **({"resample_policy": RESAMPLE_POLICY_VERSION} if derived_versions else {}),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _input_codec_for_image_format(image_format: str, topic: str) -> Literal["mjpeg", "png"]:
    """Map a CompressedImage ``format`` field to an ffmpeg input codec.

    ROS ``compressed_image_transport`` writes strings like
    ``"bgr8; jpeg compressed bgr8"``, so match by substring.
    """
    lowered = image_format.lower()
    if "png" in lowered:
        return "png"
    if "jpeg" in lowered or "jpg" in lowered:
        return "mjpeg"
    raise ValueError(
        f"camera topic {topic!r} carries unsupported compressed-image format "
        f"{image_format!r}; v1 transcodes jpeg and png"
    )


@dataclass
class _OutgoingMessage:
    log_time: int
    publish_time: int
    # Source channel id for pass-through/transcoded messages; the derived
    # topic (a str, disjoint from int ids) for derived-signal messages. Both
    # key ``output_channel_ids`` when messages are written.
    source_channel_id: int | str
    data: bytes


def _bulk_topics(
    outgoing: "Sequence[_OutgoingMessage]", infos: "Mapping[int, TopicInfo]"
) -> frozenset[str]:
    """Non-camera topics too large per message to share the state group.

    Grouping is a read-pattern decision. A training sample co-reads the
    cameras and the small state channels and never touches the point clouds,
    so a lidar topic sharing chunks with ``/imu`` makes every ``/imu`` read
    drag bulk bytes along -- measured at 230 MB versus 4 MB on real footage
    in docs/BENCHMARKS.md. Mean payload size is the proxy that separates them
    without needing to know what any particular topic means.

    Costs nothing extra: the transform has already buffered every message by
    the time this is asked, so this is arithmetic over data in hand rather
    than another pass over the input.
    """
    payload_bytes: dict[str, int] = {}
    message_counts: dict[str, int] = {}
    for message in outgoing:
        info = (
            infos.get(message.source_channel_id)
            if isinstance(message.source_channel_id, int)
            else None
        )
        if info is None or info.schema_name in CAMERA_SCHEMA_NAMES:
            # Cameras have their own group, and derived channels (a str id)
            # are grids the state group is meant for.
            continue
        payload_bytes[info.topic] = payload_bytes.get(info.topic, 0) + len(message.data)
        message_counts[info.topic] = message_counts.get(info.topic, 0) + 1
    return frozenset(
        topic
        for topic, total_bytes in payload_bytes.items()
        if total_bytes / message_counts[topic] > BULK_MESSAGE_BYTES
    )


def _group_for_topic(
    info: TopicInfo, *, bulk_topics: "frozenset[str]", topic_group_overrides: Mapping[str, str]
) -> str:
    """Which chunk group one topic's channels are written into.

    Topic-keyed on purpose: an override applies to every channel of a topic,
    and several channels may legally share one.
    """
    override = topic_group_overrides.get(info.topic)
    if override is not None:
        return override
    if info.schema_name in CAMERA_SCHEMA_NAMES:
        return DEFAULT_CAMERA_GROUP
    return DEFAULT_BULK_GROUP if info.topic in bulk_topics else DEFAULT_STATE_GROUP


def _derived_group_chunk_sizes(
    outgoing: "Sequence[_OutgoingMessage]",
    *,
    topic_group_by_message: Mapping[str, str],
    infos: "Mapping[int, TopicInfo]",
    read_window_seconds: float,
) -> dict[str, int]:
    """A chunk target per group, from the rate that group is written at.

    Groups differ by orders of magnitude -- six cameras produce megabytes a
    second where a proprio channel produces kilobytes -- so one threshold
    cannot be right for both. Sizing each to what a read of this workload
    actually spans is what makes a chunk a useful unit of fetching rather
    than an arbitrary one.

    Rate is measured over the episode's own span, which is what the file
    records; a group with no span (one message, or all messages at one
    instant) falls back to the floor rather than dividing by zero.
    """
    group_bytes: dict[str, int] = {}
    group_first_time: dict[str, int] = {}
    group_last_time: dict[str, int] = {}
    for message in outgoing:
        source_channel = message.source_channel_id
        topic = source_channel if isinstance(source_channel, str) else infos[source_channel].topic
        group_name = topic_group_by_message.get(topic)
        if group_name is None:
            continue
        group_bytes[group_name] = group_bytes.get(group_name, 0) + len(message.data)
        group_first_time.setdefault(group_name, message.log_time)
        group_first_time[group_name] = min(group_first_time[group_name], message.log_time)
        group_last_time[group_name] = max(
            group_last_time.get(group_name, message.log_time), message.log_time
        )
    derived: dict[str, int] = {}
    for group_name, total_bytes in group_bytes.items():
        span_ns = group_last_time[group_name] - group_first_time[group_name]
        if span_ns <= 0:
            derived[group_name] = MINIMUM_DERIVED_CHUNK_SIZE_BYTES
            continue
        bytes_per_second = total_bytes / (span_ns / 1e9)
        derived[group_name] = derived_chunk_size_bytes(bytes_per_second, read_window_seconds)
    return derived


def _resolve_decoder(topic: str, info: TopicInfo) -> Callable[[bytes], Any]:
    """A decoder for one topic's payloads, or a loud error naming the topic."""
    # Imported here: decoder support packages are only needed on this path.
    from mcap.records import Schema
    from mcap_protobuf.decoder import DecoderFactory as ProtobufDecoderFactory
    from mcap_ros2.decoder import DecoderFactory as Ros2DecoderFactory

    schema = Schema(
        id=1, name=info.schema_name, encoding=info.schema_encoding, data=info.schema_data
    )
    for factory in (Ros2DecoderFactory(), ProtobufDecoderFactory()):
        decoder = factory.decoder_for(info.message_encoding, schema)
        if decoder is not None:
            return decoder
    raise ValueError(
        f"camera topic {topic!r} has message encoding {info.message_encoding!r} "
        "that no available decoder handles"
    )


def _validate_passthrough_video_payload(
    topic: str, decoded_message: Any, *, is_first_message: bool
) -> None:
    """Enforce the canonical video constraints on a pass-through message.

    The provenance stamp asserts the whole file conforms (FORMAT.md), so a
    pre-encoded video channel from another recorder must be proven
    conforming, not trusted.
    """
    if decoded_message.format != "h264":
        raise ValueError(
            f"pass-through video topic {topic!r} carries format "
            f"{decoded_message.format!r}; the canonical convention requires 'h264' "
            "(re-encode upstream -- v1 does not transcode CompressedVideo)"
        )
    try:
        access_units = video_module.split_annex_b_stream(bytes(decoded_message.data))
    except ValueError as error:
        raise ValueError(
            f"pass-through video topic {topic!r} violates the canonical convention: "
            f"{error} (re-encode upstream -- v1 does not transcode CompressedVideo)"
        ) from error
    if len(access_units) != 1:
        raise ValueError(
            f"pass-through video topic {topic!r} has a message containing "
            f"{len(access_units)} access units; the canonical convention requires "
            "exactly one decodable frame per message"
        )
    access_unit = access_units[0]
    if access_unit.is_keyframe and not access_unit.has_parameter_sets:
        raise ValueError(
            f"pass-through video topic {topic!r} has a keyframe without SPS/PPS; "
            "the canonical convention requires parameter sets on every keyframe"
        )
    if is_first_message and not access_unit.is_keyframe:
        raise ValueError(
            f"pass-through video topic {topic!r} starts mid-GOP: its first message "
            "is not a keyframe, so the stream is not decodable from the start"
        )


def _decode_compressed_images(
    topic: str, info: TopicInfo, raw_payloads: list[bytes]
) -> tuple[list[bytes], list[str], str]:
    """Decode CompressedImage payloads to (image bytes, frame ids, ffmpeg codec)."""
    decoder = _resolve_decoder(topic, info)

    images: list[bytes] = []
    frame_ids: list[str] = []
    formats: set[str] = set()
    for payload in raw_payloads:
        decoded = decoder(payload)
        images.append(bytes(decoded.data))
        formats.add(str(decoded.format))
        header = getattr(decoded, "header", None)
        frame_ids.append(str(header.frame_id) if header is not None else "")
    if len(formats) != 1:
        raise ValueError(f"camera topic {topic!r} mixes image formats {sorted(formats)}")
    return images, frame_ids, _input_codec_for_image_format(formats.pop(), topic)


def write_canonical_episode(
    source: Path | str,
    output: Path | str,
    config: TransformConfig | None = None,
    *,
    source_uri: str | None = None,
    derived: Sequence[tuple[str, DerivedSeries, str]] | None = None,
) -> EpisodeStamps:
    """Transform ``source`` (any standard MCAP) into a canonical episode at
    ``output`` and return the stamps written into its provenance record.

    ``derived`` entries are ``(topic, series, version)``: each series is
    written as a new JSON channel on ``topic`` (see the module docstring).
    """
    transform_config = config if config is not None else TransformConfig()
    source_path = Path(source)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    derived_channels = list(derived) if derived is not None else []
    derived_topics = [derived_topic for derived_topic, _series, _version in derived_channels]
    if len(set(derived_topics)) != len(derived_topics):
        raise ValueError(f"derived topics must be unique, got {derived_topics}")

    gop_seconds = (
        transform_config.gop_seconds
        if transform_config.gop_seconds is not None
        else GOP_SECONDS[transform_config.gop_preset]
    )
    pipeline_version = compute_pipeline_version(
        transform_config,
        {derived_topic: version for derived_topic, _series, version in derived_channels},
    )

    reader = open_reader(source_path)
    try:
        # Keyed by CHANNEL id: several channels may legally share a topic and
        # each must survive the transform as its own output channel.
        infos = reader.channels()
        for info in infos.values():
            if info.schema_name in _RAW_IMAGE_SCHEMAS:
                raise NotImplementedError(
                    f"raw image channel {info.topic!r} ({info.schema_name}) is not supported "
                    "in v1; record compressed images upstream"
                )
        source_topic_names = {info.topic for info in infos.values()}
        for derived_topic in derived_topics:
            if derived_topic in source_topic_names:
                # A second channel on the topic would be MCAP-legal but make
                # topic-keyed reads ambiguous forever; refuse loudly instead.
                raise ValueError(
                    f"derived channel topic {derived_topic!r} collides with a source "
                    "channel topic; pick a topic not present in the source"
                )

        # The resolved layout, recorded into provenance below so it is a fact
        # about the published episode rather than a rule to re-derive. Filled
        # once, after the read, because the bulk rule needs message sizes; the
        # per-group chunk targets read the same map, so the assignment cannot
        # be computed one way for the writer and another for registration.
        resolved_topic_groups: dict[str, str] = {}

        def group_for(info: TopicInfo) -> str:
            return resolved_topic_groups[info.topic]

        transcode_channel_ids = sorted(
            channel_id
            for channel_id, info in infos.items()
            if info.schema_name in _TRANSCODABLE_IMAGE_SCHEMAS
        )

        # Read everything up front: passthrough raw bytes, plus per-camera
        # payloads to transcode. Whole episodes in memory is a deliberate v1
        # simplification (few-hours scale; see the design tenets).
        outgoing: list[_OutgoingMessage] = []
        camera_log_times: dict[int, list[int]] = {cid: [] for cid in transcode_channel_ids}
        camera_publish_times: dict[int, list[int]] = {cid: [] for cid in transcode_channel_ids}
        camera_payloads: dict[int, list[bytes]] = {cid: [] for cid in transcode_channel_ids}
        passthrough_video_channel_ids = {
            channel_id
            for channel_id, info in infos.items()
            if info.schema_name in _PASSTHROUGH_VIDEO_SCHEMAS
        }
        passthrough_video_decoders: dict[int, Callable[[bytes], Any]] = {}
        passthrough_video_channels_with_messages: set[int] = set()
        for batch in reader.iter_batches():
            if batch.channel_id in camera_payloads:
                camera_log_times[batch.channel_id].extend(int(t) for t in batch.log_times)
                camera_publish_times[batch.channel_id].extend(int(t) for t in batch.publish_times)
                camera_payloads[batch.channel_id].extend(batch.data)
            else:
                if batch.channel_id in passthrough_video_channel_ids:
                    if batch.channel_id not in passthrough_video_decoders:
                        passthrough_video_decoders[batch.channel_id] = _resolve_decoder(
                            batch.topic, infos[batch.channel_id]
                        )
                    decode = passthrough_video_decoders[batch.channel_id]
                    for payload in batch.data:
                        is_first_message = (
                            batch.channel_id not in passthrough_video_channels_with_messages
                        )
                        _validate_passthrough_video_payload(
                            batch.topic,
                            decode(payload),
                            is_first_message=is_first_message,
                        )
                        passthrough_video_channels_with_messages.add(batch.channel_id)
                outgoing.extend(
                    _OutgoingMessage(
                        log_time=int(batch.log_times[index]),
                        publish_time=int(batch.publish_times[index]),
                        source_channel_id=batch.channel_id,
                        data=payload,
                    )
                    for index, payload in enumerate(batch.data)
                )

        from foxglove_schemas_protobuf.CompressedVideo_pb2 import CompressedVideo

        for channel_id in transcode_channel_ids:
            payloads = camera_payloads[channel_id]
            topic = infos[channel_id].topic
            if not payloads:
                logger.warning("camera topic %r has no messages; dropping it", topic)
                continue
            images, frame_ids, input_codec = _decode_compressed_images(
                topic, infos[channel_id], payloads
            )
            log_times = camera_log_times[channel_id]
            if len(log_times) >= 2:
                # Raises with the topic named on duplicate/non-increasing times.
                fps = video_module.estimate_fps_from_log_times(log_times, topic=topic)
            else:
                fps = 1.0  # degenerate single-frame stream; GOP collapses to 1
            gop_frames = max(1, round(gop_seconds * fps))
            access_units = video_module.encode_images_to_h264(
                images,
                fps=fps,
                gop_frames=gop_frames,
                crf=transform_config.crf,
                input_codec=input_codec,
            )
            for index, access_unit in enumerate(access_units):
                message = CompressedVideo()
                message.timestamp.FromNanoseconds(log_times[index])
                message.frame_id = frame_ids[index] or topic.strip("/")
                message.data = access_unit.data
                message.format = "h264"
                outgoing.append(
                    _OutgoingMessage(
                        log_time=log_times[index],
                        publish_time=camera_publish_times[channel_id][index],
                        source_channel_id=channel_id,
                        data=message.SerializeToString(),
                    )
                )

        # ffmpeg is an instrument only when a transcode actually consumed it;
        # pass-through validation and state channels never resolve the binary.
        video_transcode_ran = any(payloads for payloads in camera_payloads.values())

        for derived_topic, series, _version in derived_channels:
            column_values = {
                column_name: column_array.tolist()  # native Python scalars: json-encodable
                for column_name, column_array in series.values.items()
            }
            for point_index, timestamp_ns in enumerate(series.timestamps_ns.tolist()):
                payload_object = {
                    column_name: values[point_index]
                    for column_name, values in column_values.items()
                }
                outgoing.append(
                    _OutgoingMessage(
                        log_time=timestamp_ns,
                        publish_time=timestamp_ns,
                        source_channel_id=derived_topic,
                        data=json.dumps(payload_object, sort_keys=True).encode(),
                    )
                )

        # A TOTAL order, because these bytes are the episode's identity.
        #
        # log_time alone leaves ties -- on a realistic multi-stream episode
        # roughly 40% of messages share a timestamp with another -- and a
        # stable sort then settles them by the order this function happened to
        # append them in: passthrough during the read, then transcoded video,
        # then derived channels. That is a property of the code, not of the
        # episode, so reordering any of those appends would silently change
        # every content-addressed episode_id in every corpus.
        #
        # Topic breaks the tie instead: it is semantic and comparable across
        # recordings, where a recorder-assigned channel id is neither. Within
        # one topic the sort's stability preserves the source's own order for
        # that topic, which is already well defined.
        def output_topic(message: _OutgoingMessage) -> str:
            source_channel = message.source_channel_id
            return (
                source_channel if isinstance(source_channel, str) else infos[source_channel].topic
            )

        outgoing.sort(key=lambda message: (message.log_time, output_topic(message)))

        # Every message is in hand now, so which non-camera topics are bulk is
        # arithmetic rather than another pass over the input. Resolved before
        # the writer block, which is where group_for is first called.
        bulk_topics = _bulk_topics(outgoing, infos)
        resolved_topic_groups.update(
            {
                info.topic: _group_for_topic(
                    info,
                    bulk_topics=bulk_topics,
                    topic_group_overrides=transform_config.topic_groups,
                )
                for info in infos.values()
            }
        )
        for derived_topic, _series, _version in derived_channels:
            resolved_topic_groups[derived_topic] = transform_config.topic_groups.get(
                derived_topic, DEFAULT_STATE_GROUP
            )

        from mcap_protobuf.schema import build_file_descriptor_set

        source_metadata = reader.metadata()
        episode_record_present = METADATA_RECORD_EPISODE in source_metadata
        episode_record = dict(source_metadata.get(METADATA_RECORD_EPISODE, {}))
        robot_software_version = episode_record.get(EPISODE_KEY_ROBOT_SOFTWARE_VERSION)
        stamps = EpisodeStamps(
            schema_version=EPISODE_FORMAT_VERSION,
            pipeline_version=pipeline_version,
            # Resolving ffmpeg for a file it never touched would both lie in
            # the provenance and force the pinned download on camera-less
            # episodes -- stamp the honest "not-used" literal instead.
            ffmpeg_version=ffmpeg_version() if video_transcode_ran else FFMPEG_VERSION_NOT_USED,
            robot_software_version=robot_software_version,
        )

        resolved_chunk_sizes: dict[str, int] = {}
        if transform_config.chunk_size_bytes is None:
            resolved_chunk_sizes = _derived_group_chunk_sizes(
                outgoing,
                topic_group_by_message=resolved_topic_groups,
                infos=infos,
                read_window_seconds=READ_WINDOW_SECONDS[transform_config.gop_preset],
            )
        with CanonicalMcapWriter(
            output_path,
            chunk_size=(
                resolved_chunk_sizes
                if transform_config.chunk_size_bytes is None
                else transform_config.chunk_size_bytes
            ),
            compression=transform_config.compression,
        ) as writer:
            video_schema_id: int | None = None
            # Source channel id (or derived topic) -> output writer channel
            # id. Source registration is ordered by (topic, source channel
            # id), derived registration by topic, for deterministic output.
            output_channel_ids: dict[int | str, int] = {}
            for source_channel_id in sorted(infos, key=lambda cid: (infos[cid].topic, cid)):
                info = infos[source_channel_id]
                if source_channel_id in camera_payloads:
                    if not camera_payloads[source_channel_id]:
                        continue
                    if video_schema_id is None:
                        video_schema_id = writer.register_schema(
                            name=CANONICAL_VIDEO_SCHEMA_NAME,
                            encoding="protobuf",
                            data=build_file_descriptor_set(CompressedVideo).SerializeToString(),
                        )
                    output_channel_ids[source_channel_id] = writer.register_channel(
                        info.topic,
                        message_encoding="protobuf",
                        schema_id=video_schema_id,
                        group=group_for(info),
                    )
                else:
                    if info.has_schema:
                        schema_id = writer.register_schema(
                            name=info.schema_name,
                            encoding=info.schema_encoding,
                            data=info.schema_data,
                        )
                    else:
                        # Preserve the source's "no schema" sentinel instead of
                        # fabricating an empty Schema record.
                        schema_id = 0
                    output_channel_ids[source_channel_id] = writer.register_channel(
                        info.topic,
                        message_encoding=info.message_encoding,
                        schema_id=schema_id,
                        group=group_for(info),
                    )

            derived_schema_id: int | None = None
            for derived_topic, _series, _version in sorted(
                derived_channels, key=lambda entry: entry[0]
            ):
                if derived_schema_id is None:
                    # One shared minimal object schema: the record name stays
                    # neutral and format-versioned (never the project name).
                    derived_schema_id = writer.register_schema(
                        name=DERIVED_SCHEMA_NAME,
                        encoding="jsonschema",
                        data=json.dumps({"type": "object"}).encode(),
                    )
                output_channel_ids[derived_topic] = writer.register_channel(
                    derived_topic,
                    message_encoding="json",
                    schema_id=derived_schema_id,
                    group=transform_config.topic_groups.get(derived_topic, DEFAULT_STATE_GROUP),
                )

            for message in outgoing:
                writer.write_message(
                    output_channel_ids[message.source_channel_id],
                    log_time=message.log_time,
                    data=message.data,
                    publish_time=message.publish_time,
                )

            for record_name, record_data in source_metadata.items():
                if record_name in (METADATA_RECORD_EPISODE, METADATA_RECORD_PROVENANCE):
                    continue
                writer.add_metadata(record_name, record_data)
            # Present-but-empty records copy through too: presence is source
            # information, distinct from content.
            if episode_record_present or episode_record:
                writer.add_metadata(METADATA_RECORD_EPISODE, episode_record)
            provenance: dict[str, str] = {
                PROVENANCE_KEY_SCHEMA_VERSION: stamps.schema_version,
                PROVENANCE_KEY_PIPELINE_VERSION: stamps.pipeline_version,
                "ffmpeg_version": stamps.ffmpeg_version,
                "gop_preset": str(transform_config.gop_preset),
                "gop_seconds": f"{gop_seconds:g}",
            }
            if source_uri is not None:
                provenance["source_uri"] = source_uri
            # The resolved layout, per topic. Group names live nowhere in the
            # MCAP itself, and the assignment is now partly data-derived, so
            # without this a published episode's layout could only be inferred
            # from chunk membership.
            for topic, group_name in sorted(resolved_topic_groups.items()):
                provenance[f"{PROVENANCE_KEY_TOPIC_GROUP_PREFIX}{topic}"] = group_name
            # Only when the policy actually chose them. A derived target is a
            # fact about this episode's own byte rates, so unlike the
            # configured value it cannot be read back off the config.
            for group_name, chunk_target in sorted(resolved_chunk_sizes.items()):
                provenance[f"{PROVENANCE_KEY_CHUNK_TARGET_PREFIX}{group_name}"] = str(chunk_target)
            if derived_channels:
                # The alignment policy is part of every derived channel's
                # identity: this is where format converters silently diverge.
                provenance[PROVENANCE_KEY_RESAMPLE_POLICY_VERSION] = RESAMPLE_POLICY_VERSION
                for derived_topic, _series, version in derived_channels:
                    provenance[f"{PROVENANCE_KEY_DERIVED_PREFIX}{derived_topic}"] = version
            writer.add_metadata(METADATA_RECORD_PROVENANCE, provenance)

            for attachment in reader.attachments():
                writer.add_attachment(
                    attachment.name,
                    attachment.media_type,
                    attachment.data,
                    log_time=attachment.log_time,
                    create_time=attachment.create_time,
                )
    finally:
        reader.close()

    return stamps
