# Publish browser-playable camera video

Canonical episodes hold H.264 in-band as `foxglove.CompressedVideo`
messages. Foxglove and Rerun play that directly ([inspect episodes in
Foxglove](./inspect-episodes-in-foxglove.md)); a browser `<video>` element,
a notebook, or a plain HTML page cannot. The `camera_video` enrichment
remuxes each camera stream losslessly into an MP4 artifact and records the
labels a player needs to place the footage on the episode's time axis next to
the catalog's intervals.

It is opt-in because it roughly doubles per-camera storage: the MP4 carries
the same access units the canonical file already stores.

## 1. Register the enrichment

```python
import hflow
from hflow.camera_video import CAMERA_VIDEO_VERSION, camera_video

app = hflow.App("kitchen", data_root="./data")
app.enrich(version=CAMERA_VIDEO_VERSION)(camera_video)
```

It runs in the `labels` stage like any enrichment, so it never runs on a
quarantined episode and can be selected or skipped by name (`camera_video`)
with `step_names`.

## 2. What lands in the catalog

Per camera topic, on the `camera_video` step:

| Record | Key | Meaning |
|---|---|---|
| artifact | `artifact/video:<topic>` | URI of the published MP4 |
| label | `<topic>/video_start_s` | seconds from the episode's first message (`episodes.start_ns`) to the first video frame |
| label | `<topic>/video_fps` | the constant frame rate the MP4 was muxed at (the stream's median frame interval) |
| label | `<topic>/video_frame_count` | access units muxed |

The MP4 plays at a constant rate, so video time `t` maps to log time
`start_ns + (video_start_s + t) * 1e9`. A stream with irregular stamps drifts
from log time by its jitter, which the `timestamp_regularity` check measures;
callers needing exact per-frame times read the message timestamps.

`video_start_s` is absent when the episode file records no statistics, in
which case `Episode.time_bounds` is `None` and the catalog row carries NULL
`start_ns`/`end_ns`.

## 3. Play it next to the evidence

`hflow serve` serves every cataloged artifact by name, so the MP4 streams
from

```text
GET /api/v1/episodes/{episode_id}/media/video:<topic>
```

and the same episode's `GET /api/v1/episodes/{episode_id}/timeline` returns
the recorded axis (`axis_source: "recorded"`) with each interval's
`start_s`/`end_s` relative to it. A seek bar drawn over that axis, with the
video offset by `video_start_s`, shows check intervals such as `freeze:<topic>`
or `gap:<topic>` as bands over the footage they describe.

Without the server, the artifact URI is a plain file path or bucket URL in
the `measurements` table:

```sql
SELECT key, value_text FROM measurements_latest
WHERE episode_id = ? AND key LIKE 'artifact/video:%'
```
