# HFlow frequently asked questions

Direct answers about HFlow's purpose, supported formats, infrastructure,
outputs, scale, and current release status.

**Project status:** pre-v1. These answers reflect the repository on
August 19, 2026.

## What is HFlow?

HFlow by Hebbian Robotics (YC S26) is an open source Python SDK for scalable
multimodal data pipelines in robotics and physical AI. It makes production-grade
data tooling and practices, typically developed inside large robotics data teams,
accessible to anyone. HFlow runs user-owned Python processing code around a
standard episode format, records provenance and quality evidence, and makes
corpus metadata queryable without loading the underlying recordings.

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

## What does a run produce?

A complete run can produce:

- a canonical MCAP episode with provenance metadata;
- quality measurements, timestamped observations, intervals, tags, and quarantine status;
- enrichment artifacts such as contact sheets;
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
