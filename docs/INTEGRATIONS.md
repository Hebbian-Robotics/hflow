# How HFlow fits the robotics data stack

HFlow is the processing and data-quality layer between landed multimodal
recordings and a curated dataset. It deliberately builds on standard storage,
orchestration, inspection, and query tools instead of replacing them.

```text
collection        processing and quality                 curation       delivery
-----------       -----------------------------------    ------------   --------
robot or human -> MCAP -> HFlow steps -> canonical MCAP -> Parquet SQL -> manifest
                              |                 |              |
                         Airflow 3       Foxglove/Rerun      DuckDB
```

## Component boundaries

| Tool or format | Its role | HFlow's role |
| --- | --- | --- |
| **MCAP** | Standard container for timestamped multimodal messages | Reads source episodes and writes the canonical episode convention used downstream |
| **Airflow 3** | Scheduling, task state, retries, logs, and operator UI | Generates and deploys DAGs from the same Python pipeline used in local development |
| **Foxglove and Rerun** | Interactive inspection of synchronized video and state streams | Writes standard MCAP that these tools can open without an HFlow-specific viewer |
| **Parquet** | Portable columnar storage | Stores append-only catalog facts, curated manifests, and portable dataset snapshots |
| **DuckDB** | SQL analytics over files | Queries the catalog, reports coverage, and selects manifest rows |
| **Object storage** | Durable shared storage for episodes and artifacts | Treats a local directory or bucket prefix as the data root and mirrors remote objects per worker |
| **FFmpeg** | Video encoding, decoding, and deterministic media analysis | Uses it behind batch-oriented episode accessors and built-in video checks |
| **Model endpoints** | Captions, classifications, segmentations, and other model judgments | Supplies frames, contact sheets, step scheduling, and result recording; users own clients, prompts, and models |
| **Training formats and loaders** | Model-specific sample layout, sharding, and training I/O | Stops at canonical episodes plus a manifest; conversion and training remain downstream |

## MCAP is the episode boundary

HFlow v1 accepts standard MCAP and writes standard MCAP. The
[canonical episode convention](./FORMAT.md) narrows how files are laid out so
camera video, state streams, metadata, and provenance remain together while
common readers still work unmodified.

Accepting standard MCAP is not the same as accepting every standard MCAP. v1
transcodes JPEG and PNG camera images. For H.264 already encoded as
`foxglove.CompressedVideo`, it can also prepend a missing access-unit delimiter
(AUD) without re-encoding: one message already supplies one access-unit
boundary, and every existing video byte stays untouched after the six-byte
delimiter. Messages that already contain an AUD still pass through byte for
byte. Other canonical-video violations are refused with the reason because the
provenance stamp asserts the output conforms. Run `hflow doctor` against a
source file to see the gaps before you ingest it; a missing-AUD finding is the
one video constraint the built-in transform repairs.

The convention is not a proprietary file format. Non-camera messages retain
their original schemas, encodings, payloads, and timestamps. Camera streams
use the ecosystem-standard `foxglove.CompressedVideo` schema so video stays
in-band and directly inspectable.

## Airflow is the scheduled runtime, not the programming model

HFlow steps are ordinary Python functions. During development,
`app.test(...)` runs them in-process. When scheduling is useful, HFlow packages
the same registered pipeline as Airflow 3 DAGs with mapped batches, retries,
stage profiles, and task logs.

This keeps Airflow deployment concerns out of processing functions while
preserving Airflow's operational surface. The [runtime guide](./RUNTIME.md)
documents both the included Compose workspace and bring-your-own Airflow
deployment.

## Foxglove and Rerun inspect the data

HFlow does not ship a competing episode viewer. Canonical episodes open in
Foxglove, Rerun, and conforming MCAP tooling because the writer changes layout
and conventions without changing the underlying format.

Use the [Foxglove inspection guide](./how-to/inspect-episodes-in-foxglove.md)
to connect a processed episode to its Airflow run and recorded quality
evidence.

## Parquet and DuckDB make the corpus queryable

Opening every recording is too expensive for corpus-wide questions. HFlow
therefore records episode metadata, quality evidence, tags, versions, and
artifact locations as append-only Parquet facts. DuckDB exposes those facts as
views and writes the selected rows to `manifest.parquet`.

The files remain usable without HFlow helpers: DuckDB, pandas, polars, and
other Parquet readers can read them directly. See the
[catalog and curation guide](./CATALOG.md).

For downstream tools that need media beside those facts,
[`hflow export snapshot`](./how-to/export-dataset-snapshot.md) writes a
selected snapshot as generic Parquet tables and either preserves artifact URIs
or copies their files into a self-contained directory. Renumics Spotlight and
FiftyOne are documented consumers, not dependencies or privileged formats.

## Model clients stay in user code

HFlow owns extraction, scheduling, provenance, and result recording around a
model-based step. The step owns the endpoint, credentials, model, prompt,
sampling policy, and interpretation of the response.

The [OpenAI vision guide](./how-to/call-openai-vision.md) demonstrates the
generic pattern with an ordinary client. The opt-in
[Build AI checks](./how-to/run-build-ai-evaluation.md) provide a named,
published hand-visibility and active-manipulation methodology while leaving
endpoint and model selection with each registration. The
[native-video provider protocol](./PROVIDERS.md) is an extension point for
servers whose request format contains reusable protocol knowledge.

## Training stays downstream

HFlow delivers canonical episodes and a version-pinned manifest. It does not
choose a model's tensor layout, batch sampler, augmentation policy, or training
framework. Those choices vary by model and belong in a downstream converter or
dataloader.

The [memory-mapped manifest recipe](./how-to/load-manifest-mmap.md) shows how a
large training job can consume the delivered manifest without turning HFlow
into a training framework.
