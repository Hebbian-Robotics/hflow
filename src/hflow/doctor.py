"""The conformance doctor: docs/FORMAT.md as an executable tool.

``hflow doctor <file>`` / :func:`diagnose` check a file against the
canonical-episode convention, in the spirit of ``mcap doctor``: container
integrity (CRC-validated read, summary section, chunk indexes, statistics),
the metadata records and their required stamps, chunk-group layout, per-topic
time ordering, and every in-band video constraint (h264, one AUD-delimited
access unit per message, SPS/PPS on keyframes, streams start on a keyframe).

Findings, not exceptions: the doctor accumulates everything it can observe
and reports levels. ``error`` breaks the convention; ``warning`` is legal but
deviates from the defaults this project writes (e.g. custom topic-group
assignments cannot be distinguished from accidental mixing by reading the
file alone).
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from mcap.reader import make_reader
from mcap.records import Channel, Schema

from hflow import video as video_module
from hflow.format import (
    METADATA_RECORD_EPISODE,
    METADATA_RECORD_PROVENANCE,
    PASSTHROUGH_VIDEO_SCHEMA_NAMES,
    PROVENANCE_KEY_PIPELINE_VERSION,
    PROVENANCE_KEY_SCHEMA_VERSION,
)
from hflow.storage import fetch_uri

# Cap repeated per-message findings so a broken 10k-frame stream reports a
# handful of examples plus a count, not 10k lines.
_MAX_FINDINGS_PER_CODE = 3


class DiagnosticLevel(StrEnum):
    """Doctor finding severity: ``error`` breaks the convention, ``warning`` is legal but non-default."""

    ERROR = "error"  # breaks the canonical convention (or the MCAP spec)
    WARNING = "warning"  # legal, but deviates from the defaults we write


@dataclass(frozen=True)
class Finding:
    level: DiagnosticLevel
    code: str  # stable kebab-case identifier
    message: str


@dataclass
class DoctorReport:
    path: Path
    findings: list[Finding] = field(default_factory=list)
    suppressed_counts: dict[str, int] = field(default_factory=dict)

    @property
    def conforming(self) -> bool:
        return not any(finding.level is DiagnosticLevel.ERROR for finding in self.findings)

    def summary(self) -> str:
        lines = [f"doctor: {self.path}"]
        if not self.findings:
            lines.append("  conforming: no findings")
        for finding in self.findings:
            lines.append(f"  [{finding.level}] {finding.code}: {finding.message}")
        for code, suppressed in sorted(self.suppressed_counts.items()):
            lines.append(f"  ... {code}: {suppressed} further occurrences suppressed")
        verdict = "CONFORMING" if self.conforming else "NOT CONFORMING"
        lines.append(f"  verdict: {verdict}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


class _FindingCollector:
    def __init__(self) -> None:
        self.findings: list[Finding] = []
        self.suppressed_counts: dict[str, int] = {}
        self._counts: dict[str, int] = {}

    def add(self, level: DiagnosticLevel, code: str, message: str) -> None:
        seen = self._counts.get(code, 0)
        self._counts[code] = seen + 1
        if seen < _MAX_FINDINGS_PER_CODE:
            self.findings.append(Finding(level=level, code=code, message=message))
        else:
            self.suppressed_counts[code] = self.suppressed_counts.get(code, 0) + 1


def _check_video_payload(
    collector: _FindingCollector,
    topic: str,
    message_index: int,
    video_format: str,
    payload: bytes,
    is_first_message: bool,
) -> None:
    if video_format != "h264":
        collector.add(
            DiagnosticLevel.ERROR,
            "video-format",
            f"{topic} message {message_index}: format {video_format!r}, convention requires 'h264'",
        )
        return
    try:
        picture_count = video_module.count_h264_pictures(payload)
    except ValueError as error:
        collector.add(
            DiagnosticLevel.ERROR,
            "video-invalid-slice-header",
            f"{topic} message {message_index}: {error}",
        )
        return
    if picture_count > 1:
        collector.add(
            DiagnosticLevel.ERROR,
            "video-multiple-access-units",
            f"{topic} message {message_index}: {picture_count} pictures, "
            "convention requires exactly one decodable frame per message",
        )
        return
    try:
        access_units = video_module.split_annex_b_stream(payload)
    except ValueError as error:
        collector.add(
            DiagnosticLevel.ERROR,
            "video-not-aud-delimited",
            f"{topic} message {message_index}: {error}",
        )
        return
    if len(access_units) != 1:
        collector.add(
            DiagnosticLevel.ERROR,
            "video-multiple-access-units",
            f"{topic} message {message_index}: {len(access_units)} access units, "
            "convention requires exactly one decodable frame per message",
        )
        return
    unit = access_units[0]
    if unit.is_keyframe and not unit.has_parameter_sets:
        collector.add(
            DiagnosticLevel.ERROR,
            "video-keyframe-missing-parameter-sets",
            f"{topic} message {message_index}: keyframe without SPS/PPS",
        )
    if is_first_message and not unit.is_keyframe:
        collector.add(
            DiagnosticLevel.ERROR,
            "video-stream-starts-mid-gop",
            f"{topic}: first message is not a keyframe; the stream is not decodable from the start",
        )


def _resolve_video_decoder(topic: str, channel: Channel, schema: Schema) -> Callable[[bytes], Any]:
    """Resolve either supported encoded-video payload representation."""
    from mcap_protobuf.decoder import DecoderFactory as ProtobufDecoderFactory
    from mcap_ros2.decoder import DecoderFactory as Ros2DecoderFactory

    for factory in (Ros2DecoderFactory(), ProtobufDecoderFactory()):
        decoder = factory.decoder_for(channel.message_encoding, schema)
        if decoder is not None:
            return decoder
    raise ValueError(
        f"video topic {topic!r} has message encoding {channel.message_encoding!r} "
        "that no available decoder handles"
    )


def diagnose(path: Path | str) -> DoctorReport:
    """Check one file against the canonical-episode convention."""
    file_path = fetch_uri(path)
    collector = _FindingCollector()
    report = DoctorReport(path=file_path)

    with file_path.open("rb") as stream:
        try:
            reader = make_reader(stream, validate_crcs=True)
            summary = reader.get_summary()
        except Exception as error:
            collector.add(DiagnosticLevel.ERROR, "unreadable", f"not readable as MCAP: {error}")
            report.findings = collector.findings
            return report

        if summary is None:
            collector.add(
                DiagnosticLevel.ERROR,
                "no-summary",
                "no summary section: the file is unindexed (or truncated)",
            )
            report.findings = collector.findings
            return report

        schema_names_by_channel_id = {
            channel.id: (
                summary.schemas[channel.schema_id].name
                if channel.schema_id in summary.schemas
                else ""
            )
            for channel in summary.channels.values()
        }
        topics_by_channel_id = {channel.id: channel.topic for channel in summary.channels.values()}
        video_channel_ids = {
            channel_id
            for channel_id, schema_name in schema_names_by_channel_id.items()
            if schema_name in PASSTHROUGH_VIDEO_SCHEMA_NAMES
        }

        metadata_records = {record.name: dict(record.metadata) for record in reader.iter_metadata()}
        provenance = metadata_records.get(METADATA_RECORD_PROVENANCE)

        group_by_topic = {}
        if provenance:
            for k, v in provenance.items():
                if k.startswith("group/"):
                    topic_name = k[len("group/") :]
                    group_by_topic[topic_name] = v

        if summary.statistics is None:
            collector.add(
                DiagnosticLevel.ERROR, "no-statistics", "summary has no Statistics record"
            )
        if not summary.chunk_indexes:
            collector.add(
                DiagnosticLevel.ERROR,
                "no-chunk-indexes",
                "no ChunkIndex records: the file is unchunked or unindexed",
            )
        for chunk_number, chunk_index in enumerate(summary.chunk_indexes):
            chunk_channel_ids = set(chunk_index.message_index_offsets.keys())
            if not chunk_channel_ids:
                collector.add(
                    DiagnosticLevel.ERROR,
                    "chunk-missing-message-indexes",
                    f"chunk {chunk_number} has no MessageIndex records",
                )
                continue

            if group_by_topic:
                chunk_groups = set()
                for channel_id in chunk_channel_ids:
                    topic = topics_by_channel_id[channel_id]
                    if topic in group_by_topic:
                        chunk_groups.add(group_by_topic[topic])

                if len(chunk_groups) > 1:
                    mixed_topics = sorted(
                        topics_by_channel_id[channel_id] for channel_id in chunk_channel_ids
                    )
                    collector.add(
                        DiagnosticLevel.WARNING,
                        "chunk-mixes-groups",
                        f"chunk {chunk_number} mixes groups {sorted(chunk_groups)} (channels {mixed_topics}); "
                        "the default convention separates them",
                    )
            else:
                has_video = any(channel_id in video_channel_ids for channel_id in chunk_channel_ids)
                has_state = any(
                    channel_id not in video_channel_ids for channel_id in chunk_channel_ids
                )
                if has_video and has_state:
                    mixed_topics = sorted(
                        topics_by_channel_id[channel_id] for channel_id in chunk_channel_ids
                    )
                    collector.add(
                        # A custom topic-group assignment could legally do this;
                        # the file alone cannot distinguish that from a writer bug.
                        DiagnosticLevel.WARNING,
                        "chunk-mixes-video-and-state",
                        f"chunk {chunk_number} mixes video and state channels {mixed_topics}; "
                        "the default convention separates them",
                    )

        if provenance is None:
            collector.add(
                DiagnosticLevel.ERROR,
                "missing-provenance",
                f"no {METADATA_RECORD_PROVENANCE!r} metadata record (version stamps)",
            )
        else:
            for required_key in (PROVENANCE_KEY_SCHEMA_VERSION, PROVENANCE_KEY_PIPELINE_VERSION):
                if required_key not in provenance:
                    collector.add(
                        DiagnosticLevel.ERROR,
                        "provenance-missing-key",
                        f"{METADATA_RECORD_PROVENANCE} lacks {required_key!r}",
                    )
        if METADATA_RECORD_EPISODE not in metadata_records:
            collector.add(
                DiagnosticLevel.WARNING,
                "missing-episode-record",
                f"no {METADATA_RECORD_EPISODE!r} metadata record (task/operator/success "
                "semantics live there)",
            )

        # Full message pass: CRC validation happens as a side effect of
        # reading every chunk; per-topic time order and video constraints are
        # checked message by message.
        def iter_all_messages() -> Iterator[tuple[int, int, bytes]]:
            for _schema, channel, message in reader.iter_messages(log_time_order=False):
                yield channel.id, message.log_time, message.data

        last_log_time_by_channel: dict[int, int] = {}
        video_message_counts: dict[int, int] = {}
        try:
            video_decoders: dict[int, Callable[[bytes], Any]] = {}

            for channel_id, log_time, payload in iter_all_messages():
                previous = last_log_time_by_channel.get(channel_id)
                if previous is not None and log_time < previous:
                    collector.add(
                        DiagnosticLevel.ERROR,
                        "topic-time-order",
                        f"{topics_by_channel_id[channel_id]}: log_time decreases "
                        f"({previous} -> {log_time})",
                    )
                last_log_time_by_channel[channel_id] = log_time
                if channel_id in video_channel_ids:
                    message_index = video_message_counts.get(channel_id, 0)
                    video_message_counts[channel_id] = message_index + 1
                    decoder = video_decoders.get(channel_id)
                    if decoder is None:
                        channel = summary.channels[channel_id]
                        schema = summary.schemas.get(channel.schema_id)
                        if schema is None:
                            raise ValueError(
                                f"video topic {channel.topic!r} has no readable schema record"
                            )
                        decoder = _resolve_video_decoder(channel.topic, channel, schema)
                        video_decoders[channel_id] = decoder
                    decoded = decoder(payload)
                    _check_video_payload(
                        collector,
                        topics_by_channel_id[channel_id],
                        message_index,
                        decoded.format,
                        bytes(decoded.data),
                        is_first_message=message_index == 0,
                    )
        except Exception as error:
            collector.add(
                DiagnosticLevel.ERROR,
                "read-failed",
                f"reading messages failed (corrupt chunk or bad CRC?): {error}",
            )

    report.findings = collector.findings
    report.suppressed_counts = collector.suppressed_counts
    return report
