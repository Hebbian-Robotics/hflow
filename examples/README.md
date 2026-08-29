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

**Prerequisites:** an OpenAI-compatible vision endpoint. The pinned evaluation
replay additionally needs the `hf` CLI and sufficient disk for the selected
Parquet files. Each frame makes two API calls and can incur charges.

```bash
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
export OPENAI_MODEL="qwen/qwen3-vl-32b-instruct"
export BUILD_AI_API_KEY_ENV="OPENROUTER_API_KEY"
uv run --project examples/build_ai_evaluation \
    python -m examples.build_ai_evaluation.pipeline
```

The HFlow pipeline synthesizes a sample episode when no MCAP path is supplied,
evaluates its first frame through two `@app.check` steps, and records the raw
answers, parsed predictions, routed model, and token usage as
`hflow.CheckResult` measurements. Pass an MCAP path to run the checks on an
existing episode.

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

## LeRobot Dataset v3 converter

**Use it for:** converting episodes from LeRobot Dataset v3 repositories
(Parquet + av1 MP4) into canonical MCAP episodes with H.264 video, ready for
`hflow doctor` and `App.process`.

The converter is metadata-driven: the repository's feature schema, frames per
second, episode boundaries, and video paths are read from `meta/info.json`
and the episode index, so no dataset-specific constants live in the adapter.

**Supported v3 subset:** video camera features (`dtype: video`) and
fixed-width float32 state and action vectors. Image-only features, language
tensors, depth maps, and arbitrary nested features are out of scope and fail
loud before any output is published.

**Prerequisites:** `ffmpeg` on `PATH`; network access to Hugging Face.

PushT (single camera, 2-dimensional vectors):

```bash
uv run python examples/lerobot/prepare.py \
  --repo lerobot/pusht --revision main \
  --output-dir ./data/lerobot_pusht \
  --camera-key observation.image \
  --episode-index 0
```

Multi-camera real dataset (two 640x480 cameras, six-dimensional state and
action, 30 fps, about 86 MB):

```bash
uv run python examples/lerobot/prepare.py \
  --repo lerobot/svla_so101_pickplace \
  --revision f641879e22172be7e8161d5e6c1503c2d2feb657 \
  --output-dir ./data/lerobot_svla \
  --camera-key observation.images.up,observation.images.side \
  --episode-index 0
```

`--camera-key` accepts a comma-separated list; each selected camera becomes
its own `foxglove.CompressedVideo` channel. Omit `--episode-index` to convert
every episode.

Output: one canonical MCAP per episode at
`./data/<corpus>/landing/lerobot_episode_0001.mcap`, a
`prepared-manifest.json`, and a `_lerobot_cache/` holding downloaded sources
(re-run without redownloading). Every converted episode passes `hflow doctor`.

Code: [`lerobot/prepare.py`](./lerobot/prepare.py)


## Example requirements

Keep examples small enough to read, but complete enough to execute from the
repository root. A new example must include:

- prerequisites and external side effects, including network calls or cost;
- an exact command that works in the locked uv environment;
- the observable files or terminal result produced on success; and
- a link from this catalog and from the relevant how-to guide.
