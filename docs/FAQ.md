# HFlow frequently asked questions

Direct answers about HFlow's purpose, supported formats, infrastructure,
outputs, scale, and current release status.

**Project status:** pre-v1. These answers reflect the repository on
September 6, 2026.

## What is HFlow?

HFlow is the open source Python SDK from Hebbian Robotics (YC S26) for
verifying the quality of robotics data before it trains a model. It runs
quality checks, transformations, and enrichments as pipelines over multimodal
recordings, records provenance and evidence for every episode, and makes
corpus metadata queryable without loading the underlying recordings. The
checks include Hebbian Robotics' hosted models for questions about egocentric
footage, which the SDK calls through one API.

Teams can start with the built-in quality checks, write new processing steps, or
adapt code they already use. Quality control is a common starting point, while the
same pipeline also supports transformation, enrichment, orchestration, storage,
versioning, and curation.

For the complete system design, see the
[architecture](./ARCHITECTURE.md). For a component-by-component view, see
[how HFlow fits the robotics data stack](./INTEGRATIONS.md).

## What data can HFlow process?

HFlow v1 accepts one multimodal episode per standard
[MCAP](https://mcap.dev/) file. Cameras, teleoperated robots, autonomous policy
rollouts, and other collection systems can all feed the pipeline once their
data is represented as MCAP.

The built-in transform writes one [canonical MCAP episode](./FORMAT.md): camera
streams become in-band video, non-camera channels retain their original topic,
schema, encoding, payload bytes, and timestamps, and provenance travels with
the file.

Recordings that are not MCAP get there in two ways: `hflow import lerobot`
[imports a LeRobot Dataset v3 repository](./how-to/import-lerobot-v3.md), and
any other format takes a [small converter](./how-to/write-a-converter.md) you
write once. Human egocentric capture, UMI-style handheld grippers, robot
teleoperation, and autonomous policy rollouts all fit the same episode shape.

## What data quality problems does HFlow detect?

The [built-in checks](./how-to/enable-built-in-checks.md) cover the camera and
timing faults every corpus has: black frames, frozen cameras, camera shake,
exposure, irregular or missing timestamps, keyframe intervals that make video
slow to seek, and duplicate recordings by content or media digest. Trajectory
checks cover joint discontinuities, idle stretches, and motion facts when state
streams are present.

Two opt-in checks ask questions about the footage itself: whether the wearer's
hands are in view and whether active manipulation is happening. Each is a
contract, one frame in and a fixed answer out, answered either by Build AI's
published prompts through a vision model you name or by Hebbian Robotics'
hosted implementation. Sampled over an episode, their negative answers become
`hands_absent` and `no_manipulation` intervals. See
[the Build AI checks guide](./how-to/run-build-ai-evaluation.md).

Every finding is stored as a measurement, a timestamped observation, or a
labeled interval on the episode's own time axis, so a viewer can show exactly
where in the recording each problem occurred.

## What are the hosted checks, and does my data leave my machine?

By default nothing leaves your infrastructure: the SDK, the catalog, and every
built-in check run where you run them. The hosted checks are the exception and
are opt-in. When you register one with `HFlowHostedExecution`, the SDK sends
the sampled JPEG frames that check needs to `https://api.hflow.dev` and records
the answers in your catalog alongside every other check. No API key is needed.
The service pins its prompt, model, and settings per hosted check version and
admits one request at a time per client.

## What does a run produce?

A complete run can produce:

- a canonical MCAP episode with provenance metadata;
- quality measurements, timestamped observations, intervals, tags, and quarantine status;
- enrichment artifacts such as contact sheets and, with the opt-in
  [`camera_video` enrichment](./how-to/publish-camera-video.md),
  browser-playable MP4s aligned to the episode's time axis;
- append-only Parquet catalog rows; and
- a version-pinned Parquet manifest selected with DuckDB SQL.

The [catalog and curation guide](./CATALOG.md) defines the stored tables and
shows worked queries.

## Does HFlow require Docker or Airflow?

No. `app.test(...)` runs the complete registered pipeline in-process, with no
Docker or scheduler. It is the normal development loop and is enough to try
HFlow on a local episode.

Docker and Airflow are needed only when you want the included scheduled
runtime. `hflow up` packages the same pipeline as Airflow 3 DAGs, and
`hflow ingest` triggers runs against that deployment. See the
[runtime guide](./RUNTIME.md).

## Can I test one check on one episode?

Yes. Call the check function directly with an `hflow.Episode` for the tightest
function-level loop. This skips runner behavior such as registered gates,
quarantine, and catalog recording. The [quality-check porting
guide](./PORTING.md#test-one-check-directly) shows the pattern and when to
follow it with `app.test(...)`.

## Do I have to rewrite my existing processing code?

Usually not. Your existing function can keep accepting the numpy array, MP4
path, JPEG frames, or metadata dictionary it already understands. A small
HFlow adapter extracts that input and records the function's output.

The [quality-check porting guide](./PORTING.md) shows each supported pattern
with runnable code.

## How does HFlow decide whether data is good?

HFlow records evidence rather than imposing one universal definition of
quality. Checks can emit measurements, timestamped observations, intervals, tags, and an optional
verdict. Critical failures quarantine an episode without deleting it; later
curation queries decide which thresholds and versions belong in a particular
dataset.

This keeps raw observations reusable when quality policy changes. See
[the evidence model](./PORTING.md#the-model-evidence-not-verdicts).

## Does HFlow replace Foxglove, Rerun, Airflow, or DuckDB?

No. HFlow connects standard tools rather than replacing them:

- MCAP stores each multimodal episode;
- Foxglove and Rerun inspect canonical episodes;
- Airflow schedules production runs;
- Parquet stores the catalog and manifests; and
- DuckDB queries and curates those records.

See [how HFlow fits the robotics data stack](./INTEGRATIONS.md) for the boundary
between HFlow and each tool.

## Does HFlow train robotics models?

No. HFlow ends at canonical, quality-tagged, version-stamped episodes and a
curated manifest. Training loops, policy implementations, and model-specific
dataloaders are outside this repository.

HFlow can import a supported subset of LeRobot Dataset v3 into canonical MCAP
with `hflow import lerobot`. It does not export a curated HFlow selection back
to LeRobot or run LeRobot training; HFlow's processing and curation boundaries
remain MCAP and Parquet.

## Where can HFlow store data?

The durable data root can be a local directory or, with the optional bucket
backend, an `s3://`, `gs://`, or Azure object-store location. Pipeline
processes remain stateless around that root; workers use a local mirror for
remote objects and publish durable outputs back to the store.

See [storage and durability](./ARCHITECTURE.md#storage-and-durability) and
[bucket data roots](./RUNTIME.md#bucket-data-roots---data-root-gsbucketprefix).

## Is HFlow published on PyPI?

Yes. Install the SDK with `uv add hflow`. The Hebbian Robotics project starts
at version 0.2.0; earlier 0.1.x releases under the same PyPI name belonged to
an unrelated, inactive project before the name was transferred. Follow the
[five-minute quickstart](../examples/README.md#five-minute-quickstart) for the
repository example and its expected result.

## Where should I start?

Run the [five-minute local quickstart](../examples/README.md#five-minute-quickstart).
It synthesizes a small multimodal episode and exercises the full in-process
lifecycle without Docker, external services, or robot hardware.
