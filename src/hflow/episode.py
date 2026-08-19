"""The Episode handle: the porting surface for existing QC code.

Accessors extract the input dialects existing robotics QC scripts already
expect -- numpy arrays (``channel().to_numpy()``), an MP4 path (``video()``),
JPEG frames (``frames()``), a metadata dict (``metadata``) -- so user check
functions run unchanged. Users who want none of it can open ``ep.path`` with
the raw ``mcap`` package: the file is standard MCAP.

Decoding happens here, above the batch reader seam: CDR via
``mcap-ros2-support``, protobuf via ``mcap-protobuf-support``, JSON via the
standard library.
"""

import json
import logging
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from mcap.records import Schema

from hflow import video as video_module
from hflow.ffmpeg import ffmpeg_path
from hflow.format import (
    CAMERA_SCHEMA_NAMES,
    CANONICAL_VIDEO_SCHEMA_NAME,
    METADATA_RECORD_EPISODE,
    METADATA_RECORD_PROVENANCE,
)
from hflow.reader import EpisodeReader, TopicInfo, open_reader

logger = logging.getLogger(__name__)

RawDecoder = Callable[[bytes], Any]


@dataclass(frozen=True)
class ExtractedFrame:
    """One JPEG frame extracted from a camera stream."""

    path: Path
    log_time_ns: int


def _sanitize_topic(topic: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", topic.strip("/")) or "root"


def _message_field_names(message: Any) -> list[str]:
    slots = getattr(message, "__slots__", None)
    if slots is not None:
        return list(slots)
    descriptor = getattr(message, "DESCRIPTOR", None)
    if descriptor is not None:
        return [field.name for field in descriptor.fields]
    if isinstance(message, dict):
        return list(message.keys())
    return list(vars(message).keys())


def _message_field(message: Any, name: str) -> Any:
    if isinstance(message, dict):
        return message[name]
    return getattr(message, name)


def _is_numeric_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _is_numeric_sequence(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return value.size > 0 and value.dtype.kind in "iuf"
    if isinstance(value, (list, tuple)):
        return len(value) > 0 and _is_numeric_scalar(value[0])
    return False


class ChannelData:
    """All messages of one channel: timestamps plus lazily decoded payloads."""

    def __init__(
        self,
        topic: str,
        channel_id: int,
        info: TopicInfo,
        log_times: np.ndarray,
        publish_times: np.ndarray,
        raw: list[bytes],
        decoder: RawDecoder | None,
    ) -> None:
        self.topic = topic
        self.channel_id = channel_id
        self.info = info
        self._log_times = log_times
        self._publish_times = publish_times
        self._raw = raw
        self._decoder = decoder

    def __len__(self) -> int:
        return len(self._raw)

    @property
    def timestamps(self) -> np.ndarray:
        """Log times in nanoseconds, ascending, shape (n,). Does not decode."""
        return self._log_times

    @property
    def publish_times(self) -> np.ndarray:
        """Publish times in nanoseconds, shape (n,). Does not decode.

        Unlike :attr:`timestamps`, publish times are not guaranteed ascending
        (they reflect when the application handed each message to the channel).
        """
        return self._publish_times

    @property
    def raw(self) -> list[bytes]:
        """Raw encoded message payloads. Does not decode."""
        return self._raw

    def iter_decoded(self) -> Iterator[Any]:
        """Decode messages one at a time WITHOUT caching the decoded objects.

        Use this for payload-heavy channels (video) where keeping a second,
        decoded copy of every message alive would double memory; ``messages``
        below caches for the repeated-random-access case.
        """
        self._require_decoder()
        assert self._decoder is not None
        for payload in self._raw:
            yield self._decoder(payload)

    @cached_property
    def messages(self) -> list[Any]:
        """Decoded message objects (CDR/protobuf/JSON by channel encoding)."""
        return list(self.iter_decoded())

    def _require_decoder(self) -> None:
        if self._decoder is None:
            raise ValueError(
                f"no decoder for topic {self.topic!r} "
                f"(message_encoding={self.info.message_encoding!r}, "
                f"schema_encoding={self.info.schema_encoding!r}). "
                "Supported encodings: cdr (ros2msg), protobuf, json. "
                "You can still access .raw bytes or open ep.path with the mcap package."
            )

    def _numeric_field_candidates(self) -> tuple[list[str], list[str]]:
        first = self.messages[0]
        names = _message_field_names(first)
        sequence_fields = [n for n in names if _is_numeric_sequence(_message_field(first, n))]
        scalar_fields = [n for n in names if _is_numeric_scalar(_message_field(first, n))]
        return sequence_fields, scalar_fields

    def to_numpy(self, field: str | None = None) -> np.ndarray:
        """Stack one message field into an array of shape (n_messages, ...).

        With ``field=None``, picks the field automatically: the numeric
        array field named ``position`` if present (the JointState case),
        otherwise the only numeric array field, otherwise the only numeric
        scalar field -- and raises with the candidate list when ambiguous.
        """
        if len(self._raw) == 0:
            raise ValueError(f"topic {self.topic!r} has no messages")
        if field is None:
            sequence_fields, scalar_fields = self._numeric_field_candidates()
            if "position" in sequence_fields:
                field = "position"
            elif len(sequence_fields) == 1:
                field = sequence_fields[0]
            elif len(sequence_fields) > 1:
                raise ValueError(
                    f"topic {self.topic!r} has multiple numeric array fields "
                    f"{sequence_fields}; pass field=..."
                )
            elif len(scalar_fields) == 1:
                field = scalar_fields[0]
            elif len(scalar_fields) > 1:
                raise ValueError(
                    f"topic {self.topic!r} has multiple numeric fields "
                    f"{scalar_fields}; pass field=..."
                )
            else:
                raise ValueError(
                    f"topic {self.topic!r} ({self.info.schema_name}) has no numeric "
                    f"fields; available fields: {_message_field_names(self.messages[0])}"
                )
        else:
            available = _message_field_names(self.messages[0])
            if field not in available:
                raise KeyError(
                    f"field {field!r} not in topic {self.topic!r}; available: {available}"
                )
        values = [_message_field(message, field) for message in self.messages]
        try:
            array = np.asarray(values)
        except ValueError as error:
            raise ValueError(
                f"field {field!r} of topic {self.topic!r} is ragged (per-message "
                "lengths differ); extract and align it manually from .messages"
            ) from error
        if array.dtype.kind not in "iuf":
            raise ValueError(
                f"field {field!r} of topic {self.topic!r} is not numeric (dtype {array.dtype})"
            )
        return array

    def to_arrow(self) -> Any:
        """The channel as a ``pyarrow.Table``: ``log_time_ns`` plus one column
        per primitive field (numeric/string/bool scalars and numeric lists;
        nested fields are skipped). Requires the ``arrow`` extra."""
        try:
            import pyarrow
        except ImportError as error:
            raise ImportError(
                "pyarrow is required for to_arrow(); install the 'arrow' extra"
            ) from error
        columns: dict[str, Any] = {"log_time_ns": pyarrow.array(self._log_times)}
        for name in _message_field_names(self.messages[0]):
            values = [_message_field(message, name) for message in self.messages]
            first = values[0]
            is_primitive = isinstance(first, (int, float, str, bool, np.integer, np.floating))
            is_numeric_list = _is_numeric_sequence(first)
            if not (is_primitive or is_numeric_list):
                continue
            if isinstance(first, np.ndarray):
                values = [np.asarray(value).tolist() for value in values]
            columns[name] = pyarrow.array(values)
        return pyarrow.table(columns)


class Episode:
    """Read access to one episode file (canonical or any standard MCAP)."""

    def __init__(self, path: Path | str, workdir: Path | str | None = None) -> None:
        self.path = Path(path)
        self._explicit_workdir = Path(workdir) if workdir is not None else None
        self._temp_workdir: tempfile.TemporaryDirectory[str] | None = None
        self._channel_data_by_id: dict[int, ChannelData] = {}
        # Source frame rate per camera topic, recorded by video() for frames().
        self._video_fps: dict[str, float] = {}

    @property
    def workdir(self) -> Path:
        """Where materialized artifacts (MP4s, frames) land. A temp dir by
        default; pass ``workdir=`` to keep them somewhere inspectable."""
        if self._explicit_workdir is not None:
            self._explicit_workdir.mkdir(parents=True, exist_ok=True)
            return self._explicit_workdir
        if self._temp_workdir is None:
            self._temp_workdir = tempfile.TemporaryDirectory(prefix="episode-")
        return Path(self._temp_workdir.name)

    @cached_property
    def _reader(self) -> EpisodeReader:
        return open_reader(self.path)

    def close(self) -> None:
        if "_reader" in self.__dict__:
            self._reader.close()
        if self._temp_workdir is not None:
            self._temp_workdir.cleanup()
            self._temp_workdir = None

    def __enter__(self) -> "Episode":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @cached_property
    def _decoder_factories(self) -> list[Any]:
        from mcap_protobuf.decoder import DecoderFactory as ProtobufDecoderFactory
        from mcap_ros2.decoder import DecoderFactory as Ros2DecoderFactory

        return [Ros2DecoderFactory(), ProtobufDecoderFactory()]

    def _decoder_for(self, info: TopicInfo) -> RawDecoder | None:
        if info.message_encoding == "json":
            return lambda payload: json.loads(payload.decode())
        # The decoder factories cache by schema id, so each channel needs a
        # distinct, nonzero id; the channel id (offset past the "no schema"
        # sentinel 0) is unique per file.
        schema_id = 1 + info.channel_id
        schema = Schema(
            id=schema_id,
            name=info.schema_name,
            encoding=info.schema_encoding,
            data=info.schema_data,
        )
        for factory in self._decoder_factories:
            decoder = factory.decoder_for(info.message_encoding, schema)
            if decoder is not None:
                return decoder
        return None

    @cached_property
    def topics(self) -> dict[str, TopicInfo]:
        """Channels keyed by topic name; raises when any topic has more than
        one channel (use :attr:`channels` to handle those files)."""
        return self._reader.topics()

    @cached_property
    def channels(self) -> dict[int, TopicInfo]:
        """All channels keyed by channel id -- the authoritative view; unlike
        :attr:`topics` it represents files with several channels per topic."""
        return self._reader.channels()

    @cached_property
    def metadata_records(self) -> dict[str, dict[str, str]]:
        """All MCAP Metadata records, keyed by record name."""
        return self._reader.metadata()

    @cached_property
    def metadata(self) -> dict[str, str]:
        """Episode semantics and version stamps, flattened: the ``episode/v1``
        record merged with ``provenance/v1``. See ``metadata_records`` for
        everything else."""
        merged: dict[str, str] = {}
        merged.update(self.metadata_records.get(METADATA_RECORD_EPISODE, {}))
        merged.update(self.metadata_records.get(METADATA_RECORD_PROVENANCE, {}))
        return merged

    @property
    def cameras(self) -> list[str]:
        """Topics carrying camera streams (by schema name), sorted. Derived
        from :attr:`channels`, so duplicated non-camera topics don't break it."""
        return sorted(
            {
                info.topic
                for info in self.channels.values()
                if info.schema_name in CAMERA_SCHEMA_NAMES
            }
        )

    def _resolve_channel_info(self, key: str | int) -> TopicInfo:
        if isinstance(key, int):
            info = self.channels.get(key)
            if info is None:
                raise KeyError(
                    f"channel id {key} not in episode; channel ids: {sorted(self.channels)}"
                )
            return info
        matches = [info for info in self.channels.values() if info.topic == key]
        if not matches:
            topics = sorted({info.topic for info in self.channels.values()})
            raise KeyError(f"topic {key!r} not in episode; topics: {topics}")
        if len(matches) > 1:
            ids = sorted(info.channel_id for info in matches)
            raise ValueError(
                f"topic {key!r} has {len(matches)} channels (ids {ids}); "
                f"address one by id, e.g. ep.channel({ids[0]})"
            )
        return matches[0]

    def channel(self, key: str | int) -> ChannelData:
        """All messages of one channel, with decode-on-demand accessors.

        ``key`` is a topic name or a channel id (see :attr:`channels`); a
        topic name that several channels share raises with the channel ids.
        """
        info = self._resolve_channel_info(key)
        cached = self._channel_data_by_id.get(info.channel_id)
        if cached is not None:
            return cached
        log_time_parts: list[np.ndarray] = []
        publish_time_parts: list[np.ndarray] = []
        raw: list[bytes] = []
        for batch in self._reader.iter_batches(channel_ids=[info.channel_id]):
            log_time_parts.append(batch.log_times)
            publish_time_parts.append(batch.publish_times)
            raw.extend(batch.data)
        empty = np.asarray([], dtype=np.int64)
        data = ChannelData(
            topic=info.topic,
            channel_id=info.channel_id,
            info=info,
            log_times=np.concatenate(log_time_parts) if log_time_parts else empty,
            publish_times=np.concatenate(publish_time_parts) if publish_time_parts else empty,
            raw=raw,
            decoder=self._decoder_for(info),
        )
        self._channel_data_by_id[info.channel_id] = data
        return data

    def _resolve_camera(self, camera: str | None) -> str:
        cameras = self.cameras
        if not cameras:
            raise ValueError(f"episode {self.path.name} has no camera topics")
        if camera is None:
            if len(cameras) == 1:
                return cameras[0]
            raise ValueError(f"episode has multiple cameras {cameras}; pass one explicitly")
        if camera in cameras:
            return camera
        substring_matches = [topic for topic in cameras if camera in topic]
        if len(substring_matches) == 1:
            return substring_matches[0]
        raise ValueError(f"camera {camera!r} is ambiguous or unknown; camera topics: {cameras}")

    def video(self, camera: str | None = None) -> Path:
        """Losslessly remux a camera's in-band H.264 into an MP4 file.

        Requires the canonical ``foxglove.CompressedVideo`` channel; on a
        pre-transform episode, run ``write_canonical_episode`` first. The MP4
        is cached in ``workdir``.
        """
        topic = self._resolve_camera(camera)
        info = self._resolve_channel_info(topic)
        if info.schema_name != CANONICAL_VIDEO_SCHEMA_NAME:
            raise ValueError(
                f"camera {topic!r} is {info.schema_name!r}, not "
                f"{CANONICAL_VIDEO_SCHEMA_NAME!r}: this is not a canonical episode. "
                "Transform it first (hflow.write_canonical_episode) or read the "
                "raw messages yourself via ep.channel()/the mcap package."
            )
        channel = self.channel(topic)
        fps = video_module.estimate_fps_from_log_times(channel.timestamps.tolist(), topic=topic)
        self._video_fps[topic] = fps
        output = self.workdir / f"{_sanitize_topic(topic)}.mp4"
        if output.exists():
            # Sound because write_access_units_to_mp4 replaces atomically: a
            # file at the final path is always a completed remux.
            return output

        def validated_access_units() -> "Iterator[bytes]":
            # Stream-decode instead of channel.messages: caching a decoded
            # copy of every video payload would double the episode's memory.
            for message in channel.iter_decoded():
                if message.format != "h264":
                    raise ValueError(
                        f"camera {topic!r} carries {message.format!r}, expected 'h264'"
                    )
                yield message.data

        return video_module.write_access_units_to_mp4(
            validated_access_units(),
            fps=fps,
            output=output,
        )

    def frames(
        self,
        camera: str | None = None,
        *,
        fps: float = 1.0,
        start_s: float | None = None,
        end_s: float | None = None,
    ) -> list[ExtractedFrame]:
        """Extract JPEG frames at a user-declared ``fps`` (the input dialect
        for frames-only VLM calls and frame-based checks).

        ``start_s``/``end_s`` are seconds from the start of the camera
        stream. Each frame's ``log_time_ns`` is the log time of the source
        message it was extracted from, exact to within one source frame
        interval even across recording gaps.
        """
        topic = self._resolve_camera(camera)
        mp4 = self.video(topic)
        window_start_s = start_s if start_s is not None else 0.0
        # The label must encode the exact extraction parameters (":.6f", and
        # end_s tested against None -- 0.0 is a valid bound), or two distinct
        # requests would share one cache directory.
        end_label = f"{end_s:.6f}" if end_s is not None else "end"
        label = f"{_sanitize_topic(topic)}_f{fps:.6f}_s{window_start_s:.6f}_e{end_label}"
        output_dir = self.workdir / f"frames_{label}"
        if not output_dir.exists():
            # Extract into a temp dir and rename on success: a bare directory
            # is the cache key, so a failed run must never leave one behind.
            staging_dir = self.workdir / f"frames_{label}.tmp"
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            staging_dir.mkdir(parents=True)
            command: list[str] = [str(ffmpeg_path()), "-hide_banner", "-y", "-i", str(mp4)]
            if start_s is not None:
                command += ["-ss", f"{start_s:.6f}"]
            if end_s is not None:
                command += ["-to", f"{end_s:.6f}"]
            command += ["-vf", f"fps={fps:g}", "-q:v", "2", str(staging_dir / "frame_%06d.jpg")]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            if completed.returncode != 0 or not any(staging_dir.glob("frame_*.jpg")):
                stderr_tail = completed.stderr.strip().splitlines()[-5:]
                shutil.rmtree(staging_dir)
                raise RuntimeError(
                    f"ffmpeg frame extraction produced no frames for {topic!r} "
                    f"(exit {completed.returncode}): {stderr_tail}"
                )
            staging_dir.replace(output_dir)

        # Map each extracted frame back to the source message it came from.
        # The remux is index-preserving at constant source fps, and ffmpeg's
        # fps filter emits the frame VISIBLE at each output tick (the last
        # frame at-or-before it), so time t maps to floor(t * source_fps);
        # the epsilon absorbs float error in the estimated rate.
        source_fps = self._video_fps[topic]
        source_timestamps_ns = self.channel(topic).timestamps
        frame_paths = sorted(output_dir.glob("frame_*.jpg"))
        extracted_frames: list[ExtractedFrame] = []
        for index, frame_path in enumerate(frame_paths):
            mp4_time_s = window_start_s + index / fps
            source_index = min(int(mp4_time_s * source_fps + 1e-6), len(source_timestamps_ns) - 1)
            extracted_frames.append(
                ExtractedFrame(
                    path=frame_path,
                    log_time_ns=int(source_timestamps_ns[source_index]),
                )
            )
        return extracted_frames
