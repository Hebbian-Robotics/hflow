# Run HFlow inside a worker

Use `App.process()` for one episode or `App.process_many()` for a finite batch
inside your own container, batch job, or scheduler. Both execute the same
transform, checks, gates, and enrichments as the managed runtime. They do not
start Airflow, provision compute, or require a server.

The [runnable worker example](../../examples/embedded_worker.py) imports local
video excerpts, checks them concurrently, and returns measurements while
discarding its temporary files. It requires HFlow and FFmpeg, with no model
service or API key:

```bash
uv run python examples/embedded_worker.py recording-a.mp4 recording-b.mp4 \
    --duration-s 5 --max-workers 2
```

## Choose who owns the workspace

`process_many()` defaults to `record=True`, like `process()`: it writes
canonical outputs under `<data_root>/episodes/` and appends results to the
workspace catalog. Supply `orchestrator_run_id` to associate those catalog rows
with your scheduler's run. Use a durable data root if later jobs need them.

For a stateless task, explicitly use `record=False` **and a task-local data
root**:

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from hflow import App, Stage

with TemporaryDirectory(prefix="processing-") as workspace_directory:
    app = App("worker", data_root=Path(workspace_directory))
    batch = app.process_many(
        episode_paths,
        record=False,
        stages=(Stage.SYNC, Stage.META),
        max_workers=2,
    )
    for report in batch.reports:
        # Read/copy required artifacts before the workspace is removed.
        print(report.summary())
    if batch.has_errors:
        raise RuntimeError("one or more steps failed to execute")
```

`record=False` disables catalog appends; it is **not** a no-disk or automatic
cleanup flag. Canonical episodes, sync markers, scratch files, and any enabled
media/enrichment artifacts still use the workspace. Existing canonical outputs
can be reused. Runs that omit the metadata stage can still read existing catalog
quarantine state. A fresh task-local root avoids carrying either state between
tasks.

The caller owns the workspace lifetime. `TemporaryDirectory` handles normal
returns and Python exceptions, not host loss or a forced process kill; arrange
ephemeral-disk cleanup in the deployment. Bucket inputs can also create local
download mirrors: configure `HFLOW_MIRROR_DIR` to a task-local directory before
using them. HFlow does not delete the original input or storage managed by your
checks. The FFmpeg installation/cache is separate from episode scratch storage.

An optional `output_dir` on `process_many()` changes the **batch artifact root**;
each source still gets its own identity-derived subdirectory. It does not move
the App's catalog. Do not share one source's output directory across simultaneous
batches.

## Bound concurrency and handle failures

- `max_workers` is a positive integer. It bounds concurrent episodes in this
  process, not GPUs, CPU threads used by a library, or individual checks within
  an episode. The default is one. Registered code and shared model clients must
  be safe for concurrent calls when you increase it; use separate processes or
  workers for isolation or incompatible dependencies.
- The input iterable is materialized and duplicate source identities are
  rejected before processing. Pass finite, reasonably sized batches: all
  reports are retained in memory and returned in input order.
- `on_progress` receives `ProcessManyProgress` on the calling coordinator
  thread as completed work is collected. Event input order is not guaranteed;
  each carries the original `input_index`. At most `max_workers` episodes are
  submitted at once.
- A normal check/enrichment execution error is represented in its
  `ProcessReport`; inspect `has_errors`, per-step status, and error details.
  A quality verdict of `False` is a failed quality check, not an execution
  error. Quarantine can skip later stages according to the normal engine rules.
- Configuration, source preparation, processing infrastructure, publication,
  and callback exceptions can propagate. On an exception, new submissions
  stop, queued work is cancelled where possible, and already-running episodes
  finish before the call raises. Completed work is not rolled back. There is
  no automatic retry or failure-budget policy in this API.

For durable stage accounting and error budgets, use
[`process_stage_batch`](../HOSTING.md#run-semantics-live-in-the-library). For
embedded batches, the caller decides which failures require a task retry, how
to cancel the surrounding process, and whether any partial outputs should be
kept. A completely ephemeral job can simply retry from its original inputs.

## Return results to your scheduler

`ProcessManyReport.reports` contains `ProcessReport` objects with stamps,
per-check measurements and observations, statuses, enrichments, and quarantine
tags. `report.check("registered_name")` selects one check's report. Results
remain usable in memory after workspace cleanup; paths to deleted artifacts
do not.

These Python reports are not a transport schema: they also reference registered
Python functions and local paths. Select the scalar/structured values your
application needs, serialize them, and return them through your scheduler's
normal result mechanism. Result size limits, aggregation, credentials, model
servers, and data-retention policy belong to that integration. Measurements
and observations can themselves be sensitive data.

`App.test()` and `App.test_many()` remain development conveniences, with
test-run directories and non-recording defaults. The `TestReport`,
`TestManyReport`, and `TestManyProgress` names remain aliases of the corresponding
production report types.

## Import a video before processing

Use the production `import_video_episode()` API instead of a fixture adapter:

```python
from hflow import VideoImportConfig, import_video_episode

episode_path = import_video_episode(
    "recording.mp4",
    "task-inputs/episode.mcap",
    VideoImportConfig(
        source_start_s=30,
        duration_s=5,
        image_hz=10,
        camera_name="camera",
        metadata=(("task", "sorting"),),
    ),
)
```

The importer reads a local video and emits an input-shaped MCAP for the normal
canonical transform. It selects the first video stream, resamples the requested
excerpt to a fixed image rate, and letterboxes to the configured dimensions;
it is not a lossless video archival operation. The default image dimensions are
640 × 360. The stream must declare its duration (as MP4 streams do); sources
without a known duration are refused rather than silently truncating an excerpt.
Timestamp zero is the excerpt start unless `start_time_ns` is supplied;
the importer does not infer an absolute recording time. It does not invent
task, operator, or success labels. Caller metadata and importer provenance are
recorded separately.

The output is published only after a successful import, existing destinations
are refused, and temporary conversion files are removed on failure. Downloading
objects, selecting shards, and retaining originals remain the caller's job.
For generated fixtures and injected faults, keep using `hflow.testing`.
