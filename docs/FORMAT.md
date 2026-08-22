# Canonical MCAP episode format

**Version 1**: a convention for training-ready robot episodes in standard [MCAP](https://mcap.dev/spec).

A *canonical episode* is one MCAP file, one episode: cameras stored as in-band H.264 video, state streams preserved verbatim, semantics and version stamps carried in-file. It is what the HFlow transform writes and what the rest of the pipeline (checks, catalog, curation) consumes. This page is normative: a third party should be able to implement a conforming writer from it without reading HFlow's code.

Two load-bearing ideas here -- GOP length matched to the read pattern, and topic-group
chunking -- are measured at million-hour scale in Dyna Robotics'
[Training Dyna-2 at million-hour scale, repeatably](https://www.dyna.co/research/dyna-2-infrastructure);
the sections below cite the article where a mechanism or measurement comes from it. Everything
else (the video schema, where stamps live, the codec, the numbers) is HFlow's own choice.

The single overriding rule: **a canonical episode is spec-conforming MCAP.** Every convention here constrains *how* the file is written, never *what format* it is. Any conforming MCAP reader reads these files unmodified.

## File container

- MCAP magic, `Header`, data section, `DataEnd`, summary section, `Footer`, closing magic, per the [MCAP spec](https://mcap.dev/spec).
- `Header.profile` is the empty string `""` (the file mixes protobuf video channels with pass-through channels of arbitrary encoding, so no single profile applies).
- `Header.library` is informational only (e.g. `hflow episode-format/1 transform-behavior/1`). It deliberately carries no release number: the header is inside the bytes the content episode id hashes, so a release would otherwise give a byte-identical input a new identity. No reader may key behavior off it (see [Identifier rules](#identifier-rules)).
- Chunks are compressed with **zstd** by default (`"none"` is permitted). Each `Chunk` record carries `uncompressed_crc`; the `Footer` carries a summary CRC.
- The summary section repeats all `Schema` and `Channel` records and contains `Statistics`, all `ChunkIndex` records, `AttachmentIndex`/`MetadataIndex` records, and `SummaryOffset` records. A canonical episode always has a complete summary; unindexed files are not canonical.

## Topic-group chunking

Default MCAP writing gives each topic its own chunks, so one training sample costs a read per
topic. A canonical episode instead groups topics that share a read pattern into shared,
time-major chunk streams, so a sample costs one read per *group*; Dyna's article measured this
layout change at ~3.4× fewer chunk fetches and ~2.9× faster reads at their scale.

**The rule**: every channel is assigned to exactly one named *group*. Messages from channels in different groups MUST NOT share a chunk. Within a group, messages are written time-major (ascending `log_time` across all of the group's channels interleaved), and the group's chunks form their own sequence with non-decreasing time ranges.

| Convention | Value |
|---|---|
| Default groups | `cameras` (topics whose schema is a camera schema; see below), `state` (everything else) |
| Group override | Per-topic, at transform configuration time (`TransformConfig.topic_groups`) |
| Target uncompressed chunk size | 800 000 bytes per group (configurable; measured defaults are a benchmark-report deliverable) |
| Camera schemas | `foxglove.CompressedVideo`, `foxglove.CompressedImage`, `foxglove.RawImage`, `foxglove_msgs/msg/CompressedVideo`, `sensor_msgs/msg/CompressedImage`, `sensor_msgs/msg/Image` |

Writer mechanics (how HFlow's `CanonicalMcapWriter` does it; equivalent layouts are conforming as long as the rule above holds):

- One chunk buffer per group. A message routes to its channel's group buffer; when a group's uncompressed buffer exceeds the chunk-size target, that group's chunk is finalized. At `finish()`, every group flushes its remaining buffer.
- Each `Chunk` record is followed immediately by its `MessageIndex` records (one per channel present in that chunk), and a `ChunkIndex` in the summary points at both. Consequence: the set of channel IDs in any chunk's `message_index_offsets` maps to exactly one group; this is the property conformance checks assert.
- `Schema` and `Channel` records are written *unchunked* in the data section at registration time and repeated in the summary (spec-legal; readers that only consult the summary see them either way).

Chunking changes write order, not the format.

## In-band video

The transform re-encodes per-frame JPEG cameras to in-band H.264 with GOP length matched to
the read pattern, without giving up native visualization (Dyna's article reports ~68% storage
reduction from the same move at their scale). How the H.264 sits in the file is HFlow's own
convention:

Camera streams are messages of [`foxglove.CompressedVideo`](https://docs.foxglove.dev/docs/sdk/schemas/compressed-video), protobuf-encoded:

- `Schema`: name `foxglove.CompressedVideo`, encoding `protobuf`, data = the serialized `FileDescriptorSet` for the message.
- `Channel`: `message_encoding` `protobuf`, topic = the source camera topic, unchanged.

| Field | Contents |
|---|---|
| `timestamp` | The frame's capture time; equals the MCAP message `log_time` |
| `frame_id` | The source image's `header.frame_id`; if absent, the topic name without its leading `/` |
| `data` | Exactly one H.264 access unit (one decodable frame), Annex B format |
| `format` | `"h264"` |

The H.264 bitstream constraints (all MUST):

1. **Annex B** byte stream (start-code delimited), one complete access unit per message.
2. Every access unit **begins with an access-unit delimiter** (AUD, NAL type 9); this is what makes message boundaries recoverable from the raw stream.
3. **SPS and PPS attached to every keyframe**: each IDR access unit contains its parameter-set NALs (x264 `repeat-headers=1`), so decoding can start at any keyframe message without out-of-band state.
4. **No B-frames** (`bframes=0`): decode order equals presentation order equals message order, and frame *i* of the source maps to access unit *i*.
5. **Fixed GOP** (`keyint = min-keyint`, `scenecut=0`): keyframes land exactly every `gop_frames` messages, so seek cost is uniform and predictable.
6. 4:2:0 chroma (`yuv420p`), the universally decodable baseline.

### GOP presets

GOP length is effectively a training hyperparameter, as Dyna's article observes. The writer keys
it to how the data is read:

| Preset | Keyframe interval | Read pattern it serves |
|---|---|---|
| `vla` | 1.0 s | Short sparse windows (a keyframe seek per sample, so keep it cheap) |
| `world_model` | 6.0 s | Long contiguous sequences (keyframe cost amortizes, so favor compression) |

`gop_frames = max(1, round(gop_seconds × fps))`, with `fps` measured from the source stream (`1e9 / median(Δ log_time)`). The seconds values are configurable defaults pending the benchmark report; the provenance record (below) stamps what was used.

## State channels

Every non-camera channel passes through **byte-for-byte**: original topic name, original schema (name, encoding, data), original message encoding, original payload bytes, original `log_time`/`publish_time`. A canonical episode is therefore a *superset* format: whatever the robot recorded (ROS 2 CDR, protobuf, JSON) is still exactly there, in `state`-group chunks. Messages across all channels are written in global `log_time` order (the writer distributes them to per-group chunk streams).

Raw uncompressed image channels (`sensor_msgs/msg/Image`, `foxglove.RawImage`) are not accepted by the v1 transform; record compressed images upstream.

## Metadata records

Version stamps and episode semantics live in MCAP `Metadata` records, keeping the episode
self-contained and viewer-inspectable. The stamp requirement is the one Dyna's article states:
"Every processed episode also carries a stamp of what produced it: the schema version, the
ingestion pipeline version, and the software version running on the robot when it was recorded."
All values are strings.

### `episode/v1`: what this episode is

| Key | Meaning |
|---|---|
| `task` | What the robot was doing (e.g. `fold_napkin`) |
| `operator` | Who/what collected it |
| `success` | Collector-labeled outcome, `"true"`/`"false"` |
| `embodiment` | Robot/platform identifier |
| `robot_software_version` | Software running on the robot at record time |
| *(any user key)* | Additional semantics pass through untouched |

All keys are optional; the record is copied/merged from the source recording. Recorders are encouraged to write it at collection time.

### `provenance/v1`: what produced this file

| Key | Meaning |
|---|---|
| `schema_version` | This format's version: `"1"` |
| `pipeline_version` | Content hash (12 hex chars, SHA-256 prefix) of the producing transform configuration; two files with equal `pipeline_version` were produced by identical configuration |
| `ffmpeg_version` | Version line of the encoding instrument |
| `gop_preset` | `vla` or `world_model` |
| `gop_seconds` | The keyframe interval actually used |
| `source_uri` | Where the source recording came from (optional) |

The corpus is assumed permanently mixed-version: curation pins or excludes exact versions (they are content hashes, so there is no ordering to range over) rather than expecting uniformity. A rewrite of an episode replaces `provenance/v1`; all other source `Metadata` records are copied through unchanged.

## Attachments

MCAP `Attachment` records carry episode-scoped files that are not message streams: calibration files, URDFs. The transform copies all source attachments through (name, media type, timestamps, bytes).

## Identifier rules

Stored-data identifiers (metadata record names, their keys, group names) are **neutral and format-versioned**: they never embed a project name, and a breaking change to a record's meaning bumps its `/v1` suffix rather than mutating existing files. Topic names are the source recording's own. The only place a project name may appear is the informational `Header.library` field, which no reader may depend on.

## Conformance

A file claiming this convention must satisfy, in increasing strictness:

1. **It is valid MCAP**: readable by the stock [`mcap` package](https://mcap.dev/docs/python/) with CRC validation on, with a complete summary section.
2. **It opens in the standard viewers**: [Foxglove](https://foxglove.dev/) and [Rerun](https://rerun.io/). Opening in both is the acceptance test for every file the pipeline writes.
3. **Chunk purity**: for every `ChunkIndex`, the channel IDs in `message_index_offsets` belong to exactly one group; each group's chunk sequence is time-ordered.
4. **Video constraints**: every `foxglove.CompressedVideo` message is one Annex B access unit beginning with an AUD; every IDR access unit contains SPS and PPS; no B-frames; `format="h264"`.
5. **Stamps**: `provenance/v1` present with `schema_version` and `pipeline_version`.

`hflow doctor <file.mcap> [more.mcap ...]` executes these checks against every
file given, printing one report each, and exits with status 0 when all files
conform or 1 when any reports violations.

## References

- Dyna Robotics, [Training Dyna-2 at million-hour scale, repeatably](https://www.dyna.co/research/dyna-2-infrastructure)
  (inspiration for this release; cited above where a mechanism or measurement comes from it)
- [MCAP specification](https://mcap.dev/spec) and [Python libraries](https://mcap.dev/docs/python/)
- [foxglove.CompressedVideo schema](https://docs.foxglove.dev/docs/sdk/schemas/compressed-video)
- [Architecture](./ARCHITECTURE.md): the full design this format serves
- [How HFlow fits the robotics data stack](./INTEGRATIONS.md)
