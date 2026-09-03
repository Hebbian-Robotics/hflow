# HFlow runnable examples

Examples are executable documentation. Each one names its prerequisites, gives
one command to run, and says what successful output looks like.

Most examples use the root HFlow environment. An example with a substantial or
specialized dependency stack has its own `pyproject.toml` and tests and is a uv
workspace project. Run one from the repository root with:

```bash
uv sync --locked --project examples/path_to_example
uv run --project examples/path_to_example python examples/path_to_example/main.py
```

These projects share the repository lockfile and use the workspace copy of
HFlow, but their dependencies are not installed by a normal root `uv sync`.

## Five-minute quickstart

**Use it for:** seeing the complete in-process lifecycle on a small multimodal
episode.

**Prerequisites:** the normal development environment from the repository root;
no Docker, scheduler, network service, or API key.

```bash
uv sync --locked
uv run python examples/quickstart.py
```

The example synthesizes `data/sample/episode_0001.mcap`, writes a canonical
episode under `data/test-runs/`, and prints the measurements from timestamp,
motion, and camera-health checks. Pass an existing MCAP path as the first
argument to process your own recording.

Code: [`quickstart.py`](./quickstart.py)

## Recommended real-episode evaluation

**Use it for:** seeing HFlow's default deterministic checks and hosted semantic
checks work together on real egocentric footage.

**Prerequisites:** the normal development environment, the `hf` CLI, and
network access. The hosted checks require no account, API key, or model server.

```bash
mkdir -p data/episode-evaluation
downloaded_sample_mcap_path="$(
    hf download LightwheelAI/EgoDemo \
        'EgoStand/mcap/Thimble Removal/a8d29fea-3cf9-47f7-ad4b-4b1d0ecb7a71.mcap' \
        --repo-type dataset \
        --quiet
)"
cp "$downloaded_sample_mcap_path" data/episode-evaluation/sample.mcap

uv run python examples/evaluate_episode.py \
    data/episode-evaluation/sample.mcap \
    --camera /sensor/camera/head_left/video
```

The report includes the six automatic recording-integrity checks plus hosted
hand visibility and active manipulation. The sample's first left-camera frame
is a clear positive case, so the model evidence should report two visible hands
and active manipulation. Outputs land under `data/episode-evaluation/`.

Guide: [Evaluate your first egocentric episode](../docs/tutorials/evaluate-an-egocentric-episode.md)

Code: [`evaluate_episode.py`](./evaluate_episode.py)

## OpenAI vision check

**Use it for:** sampling an episode, making a contact sheet, and calling an
OpenAI vision model from an HFlow check.

**Prerequisites:** an OpenAI API key and the optional `openai` dependency. This
example sends image data to the configured endpoint and may incur API charges.

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL="gpt-5.4-mini"
uv run --extra openai python examples/openai_vision/pipeline.py
```

The example synthesizes a small episode when no MCAP path is provided and runs
two contact-sheet checks through the Responses API: an activity description,
and a hand-visibility fraction (missing or occluded hand positions, a quality
issue Dyna's article describes, answered as model evidence). Each check is one
API call;
`OPENAI_BASE_URL` can point the same client at another endpoint that
implements the Responses API.

Guide: [Call an OpenAI vision endpoint from a step](../docs/how-to/call-openai-vision.md)  
Code: [`openai_vision/pipeline.py`](./openai_vision/pipeline.py)

## Build AI single-frame vision checks

**Use it for:** applying Build AI's single-frame hand-count and
active-manipulation methodology as HFlow checks, then replaying their
Egocentric-10K or Egocentric-100K inputs to compare models and prompts.

**Prerequisites:** network access and the `hf` CLI for the sample recording.
The default pipeline route uses HFlow's hosted checks without an account or API
key. The pinned evaluation replay additionally needs sufficient disk for the
selected Parquet files. Each frame makes two external check calls; a configured
model provider may charge for them.

```bash
mkdir -p data/build-ai-evaluation
downloaded_sample_mcap_path="$(
    hf download LightwheelAI/EgoDemo \
        'EgoStand/mcap/Thimble Removal/a8d29fea-3cf9-47f7-ad4b-4b1d0ecb7a71.mcap' \
        --repo-type dataset \
        --quiet
)"
cp "$downloaded_sample_mcap_path" data/build-ai-evaluation/sample.mcap

uv run --project examples/build_ai_evaluation \
    python -m examples.build_ai_evaluation.pipeline \
    data/build-ai-evaluation/sample.mcap
```

The HFlow pipeline evaluates the existing episode's first frame through two
`@app.check` steps and records the raw answers, parsed predictions, execution
identity, and any available model usage as `hflow.CheckResult`
measurements. An episode path is required because generated test-pattern
footage cannot meaningfully demonstrate either judgment.

To use a local OpenAI-compatible model server instead of HFlow's hosted checks:

```bash
export BUILD_AI_EXECUTION="openai-compatible"
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_MODEL="Qwen/Qwen3-VL-8B-Instruct"

uv run --project examples/build_ai_evaluation \
    python -m examples.build_ai_evaluation.pipeline \
    data/build-ai-evaluation/sample.mcap
```

The companion adapter streams the published Parquet images into Inspect AI
without transcoding them, using the same prompts, schemas, and parsers:

```bash
uv run --project examples/build_ai_evaluation \
    python examples/build_ai_evaluation/evaluate.py run \
    --dataset 10k --source build --limit 1 --download \
    --api-key-env OPENROUTER_API_KEY
```

Inspect writes structured per-sample logs under
`data/build-ai-evaluation/runs/`; the adapter prints hand-visibility and
active-manipulation prevalence plus agreement with Build AI's published Gemini
labels and writes a compact comparison summary.

Guide, exact reproduction commands, and output contract:
[Compare vision models on the Build AI evaluations](../docs/how-to/run-build-ai-evaluation.md)

- HFlow pipeline: [`build_ai_evaluation/pipeline.py`](./build_ai_evaluation/pipeline.py)
- Reproduction adapter: [`build_ai_evaluation/evaluate.py`](./build_ai_evaluation/evaluate.py)

## EgoSuite projected-hand evaluation

**Use it for:** comparing image-only VLM hand counts with labels derived from
Lightwheel EgoSuite's synchronized 3D hand joints and head-camera calibration.

**Prerequisites:** accepted access to `LightwheelAI/EgoDemo`, the `hf` CLI,
ffmpeg, and an OpenAI-compatible vision endpoint. Dataset downloads consume
local disk, and model calls send selected images to the configured endpoint and
may incur charges.

Apply the methodology to one annotated episode as an HFlow check:

```bash
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
export OPENAI_MODEL="google/gemma-4-26b-a4b-it"
export EGOSUITE_API_KEY_ENV="OPENROUTER_API_KEY"
uv run --project examples/egosuite_evaluation \
    python -m examples.egosuite_evaluation.pipeline path/to/annotated-episode.mcap
```

The pipeline records reference and predicted class counts, output validity,
agreement, token usage, and timestamped disagreement intervals as one HFlow
check result. It defaults to ten one-per-second frames per episode; all
sampling and endpoint settings are configurable through `EGOSUITE_*`
environment variables. Set `EGOSUITE_LABEL_MANIFEST` to a label report written
by `evaluate.py labels` when multiple HFlow episode runs must use one exact,
predeclared dataset slice; the manifest digest becomes part of the check
version.

The companion Inspect adapter is for dataset-level model comparisons. First
calculate labels without calling a model:

```bash
uv run --project examples/egosuite_evaluation \
    python examples/egosuite_evaluation/evaluate.py labels \
    data/egosuite-evaluation/datasets/EgoDemo \
    --episode-count 100 --frame-stride 30 --samples-per-episode 10
```

Then run the same selected frames through a VLM:

```bash
uv run --project examples/egosuite_evaluation \
    python examples/egosuite_evaluation/evaluate.py run \
    data/egosuite-evaluation/datasets/EgoDemo \
    --episode-count 100 --frame-stride 30 --samples-per-episode 10 \
    --api-key-env OPENROUTER_API_KEY
```

For every selected frame, the example inverts the labeled camera pose,
projects both sets of 21 world-space hand joints through the camera intrinsic
matrix, and counts a hand when at least one joint lands inside the image. The
VLM receives only the extracted JPEG. Inspect AI writes its structured logs
and dataset-weighted, per-class, macro-agreement, and confusion summaries under
`data/egosuite-evaluation/runs/`. A separately declared
`--samples-per-hand-count` slice can diagnose rare classes without presenting
constructed class proportions as the dataset's natural distribution.

Guide, download command, label contract, and interpretation:
[Evaluate VLM hand counts against EgoSuite joint labels](../docs/how-to/run-egosuite-hand-evaluation.md)

- HFlow pipeline: [`egosuite_evaluation/pipeline.py`](./egosuite_evaluation/pipeline.py)
- Evaluation adapter: [`egosuite_evaluation/evaluate.py`](./egosuite_evaluation/evaluate.py)

## Egocentric factory corpus

**Use it for:** a complete corpus workflow on real data, covering Hugging Face
download, declared fault injection, local processing, Airflow scheduling,
inspection with Foxglove and DuckDB, and SQL curation.

**Prerequisites:** accepted Hugging Face dataset terms, the `hf` CLI
authenticated (`uv tool install -U huggingface_hub`, then
`hf auth login`), network access, and about 2.5 GB of disk (a ~1 GB pinned
download plus the generated corpus). Docker is needed only for the Airflow
path.

```bash
uv run python examples/egocentric/prepare.py
uv run python examples/egocentric/pipeline.py data/egocentric/landing/*.mcap
```

The preparation step writes 96 deterministic episodes, six with camera faults
declared in the tracked manifest (and editable there). The pipeline accepts
the 90 healthy episodes, quarantines the six faulty ones, and records
contact-sheet artifacts for the accepted set.

Guide and expected artifacts: [`egocentric/README.md`](./egocentric/README.md)

## Synthetic stress corpus

**Use it for:** a repeatable throughput measurement over a corpus large enough
to be worth timing, with no download.

**Prerequisites:** the normal development environment plus `ffmpeg` on `PATH`;
no Docker, network, or API key. Two hundred episodes take a few minutes and
about 1 GB of disk.

```bash
uv run python examples/stress/synthetic.py --episodes 200 --seed 42
```

The example plans a seeded corpus (2 to 10 s, one to three cameras at 10, 15,
or 30 Hz, with black segments and joint jumps injected at a fixed rate), writes
it to `./stress_corpus`, runs `App.process` over every episode, and prints the
per-episode wall time and the distribution of check statuses. The same
`--episodes` and `--seed` always plan the same corpus, so runs before and after
a change are comparable. Use `--generate-only` to build a corpus without
ingesting it, and `--ingest-only` to re-run against one you already have.

Code: [`stress/synthetic.py`](./stress/synthetic.py)

## LeRobot Dataset v3 import

**Use it for:** converting episodes from LeRobot Dataset v3 repositories
(Parquet + av1 MP4) into canonical MCAP episodes with H.264 video, ready for
`hflow doctor` and `App.process`.

The importer is metadata-driven: the repository's feature schema, frames per
second, episode boundaries, and video paths are read from `meta/info.json`
and the episode index, so no dataset-specific constants live in the adapter.

**Supported v3 subset:** video camera features (`dtype: video`) and
fixed-width float32 state and action vectors. Image-only features, language
tensors, depth maps, and arbitrary nested features are out of scope and fail
loud before any output is published.

**Prerequisites:** network access to Hugging Face and enough local disk for the
selected Parquet and video files. HFlow uses its managed FFmpeg build; no
LeRobot, PyTorch, or Hugging Face SDK installation is required.

PushT (single camera, 2-dimensional vectors):

```bash
uv run hflow import lerobot \
  --repo lerobot/pusht --revision main \
  --output-dir ./data/lerobot_pusht \
  --camera observation.image \
  --episode-index 0
```

Multi-camera real dataset (two 640x480 cameras, six-dimensional state and
action, 30 fps, about 86 MB):

```bash
uv run hflow import lerobot \
  --repo lerobot/svla_so101_pickplace \
  --revision f641879e22172be7e8161d5e6c1503c2d2feb657 \
  --output-dir ./data/lerobot_svla \
  --camera observation.images.up \
  --camera observation.images.side \
  --episode-index 0
```

Repeat `--camera` for each selected camera; each becomes its own
`foxglove.CompressedVideo` channel. A comma-separated `--camera-key` remains
accepted by the compatibility wrapper. Omit `--episode-index` to import every
episode.

Output: one canonical MCAP per episode at
`./data/<corpus>/landing/lerobot_episode_0001.mcap`, a
`prepared-manifest.json`, and a `_lerobot_cache/` holding downloaded sources
(re-run without redownloading). Every converted episode passes `hflow doctor`.

- Guide: [Import a LeRobot Dataset v3 repository](../docs/how-to/import-lerobot-v3.md)
- Public API: [`hflow.importers.lerobot`](../src/hflow/importers/lerobot.py)
- Compatibility wrapper: [`lerobot/prepare.py`](./lerobot/prepare.py)

### Exporting a curated selection back to LeRobot Dataset v3

`lerobot/export.py` turns an HFlow curation selection (a `hflow curate`
manifest parquet, or a SQL query over the catalog) into a loadable local
LeRobot Dataset v3 repository containing exactly the selected episodes.
Episode provenance stamped by the converter (`source_dataset`,
`source_revision`, `source_episode_index` in `episode/v1` metadata) is
resolved back to the source; the exporter copies the source video chunks
byte-for-byte (each source (chunk, file) keeping its identity in the
destination path) and slices the selected data rows from the source chunk
parquets, so cameras, feature schema, dtypes, shapes, and frame timing
match the source. Episode indexes are renumbered sequentially in selection
order. The exporter drives the public `hflow import lerobot` entry point to
materialize the source archive (meta, data chunks, video chunks) and copies
those chunks byte-for-byte. A dataset-card draft (`README.md`) and
`export-provenance.json` record the source repository, source commit,
exporter version, selection SQL/manifest, and selected source episode
indexes.

From a curation manifest:

```
uv run python examples/lerobot/export.py \
  --manifest ./manifest.parquet \
  --destination ./data/curated_lerobot \
  --camera-keys observation.images.up,observation.images.side
```

From a SQL query over the catalog:

```
uv run python examples/lerobot/export.py \
  --sql "SELECT episode_id, metadata_json FROM episodes_latest WHERE status = 'ok'" \
  --destination ./data/curated_lerobot \
  --camera-keys observation.images.up,observation.images.side
```

Failures abort before any output is published: mixed source repositories
or revisions, non-immutable revisions, missing provenance, missing source
episodes, duplicate selections, and a staged dataset that fails structural
or (when `lerobot` is installed) official-API validation never replace a
previously valid destination.

Code: [`lerobot/export.py`](./lerobot/export.py)


## Example requirements

Keep examples small enough to read, but complete enough to execute from the
repository root. A new example must include:

- prerequisites and external side effects, including network calls or cost;
- an exact command that works in the locked uv environment;
- the observable files or terminal result produced on success; and
- a link from this catalog and from the relevant how-to guide.
