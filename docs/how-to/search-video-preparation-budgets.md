# Search model-input video preparation budgets

Use this example when you want a coding agent to explore frame rate and resolution
while keeping the source cohort and evaluator fixed. HFlow prepares model inputs;
your agent chooses experiments. The search policy is example code, outside the
SDK's pipeline, scheduling, storage and curation APIs.

The example uses the existing
[direct model-video preparation API](../VIDEO_PRIMITIVES.md#direct-model-video-preparation).
It writes the same model-input pixels as the video importer and canonical SYNC
with matching transform settings, without an intermediate MCAP. No canonical
transform behavior or encoding default changes are required.

## Run the CPU smoke demonstration

From the repository root:

```bash
uv sync --locked --project examples/video_preparation_autoresearch
uv run --locked --project examples/video_preparation_autoresearch \
  python examples/video_preparation_autoresearch/evaluate.py demo \
  --output data/video-preparation-demo
```

Use a fresh output directory. This writes synthetic source MP4s, prepared MP4s,
development reports, a selection receipt and confirmation results. It requires
FFmpeg/ffprobe and the optional PyAV video dependency supplied by the example's
workspace project. HFlow may download its managed FFmpeg build; set
`HFLOW_FFMPEG` and `HFLOW_FFPROBE` to absolute installed executable paths to
use your own tools. No model weights, model service, credentials or GPU is used.

The source generator creates eight development episodes with left/right marker
motion. Motion begins late, so a sparse preparation can lose the temporal signal.
A fixed color-based vision function measures direction from prepared pixels;
it cannot infer direction from a single frame. Unassessable predictions count
against accuracy, with the complete episode count retained.

`demo` evaluates four explicit frame/resolution budgets. It selects the smallest
pixel-frame workload achieving 100% development accuracy, writes `selection.json`,
then evaluates that frozen budget on eight separately generated confirmation
episodes. Inspect `summary.json`, each `trial-*/report.json`, and
`confirmation/report.json`. A partial directory without `report.json` is not a
completed result. Existing output is never overwritten.

This is a bounded configuration sweep to exercise the mechanics, not an
autonomous AI research run. It demonstrates actual preparation and measurement
on synthetic media; it does not estimate VLM, VLA or real-world robot quality.

## Give the loop to a coding agent

Point your agent at
[`program.md`](../../examples/video_preparation_autoresearch/program.md).
It may edit only
[`candidate.py`](../../examples/video_preparation_autoresearch/candidate.py).
The evaluator, sources, detector and encoding settings stay fixed. No agent
provider or paid agent calls are made by the example itself.

The provided instructions specify a baseline, one hypothesis per trial,
input-budget ranking subject to a fixed quality requirement, an eight-trial
limit, and separate selection and confirmation commands. The `freeze` command
checks that the development report matches the current candidate and evaluator.
Confirmation reads the frozen budget, verifies evaluator and report hashes, and
does not read an edited candidate configuration.

These checks catch accidental evidence changes. They do not sandbox the agent,
hide confirmation seeds, authenticate reports or prevent a user from editing
both evidence and hashes. For a real experiment, keep reserved recordings and
labels outside the agent's accessible filesystem. Once confirmation is opened,
those inputs are exposed and must not be reused as fresh confirmation next time.

## Interpret the measurements

Reports retain source/prepared hashes, the complete preparation budget and
encoding settings, evaluator identity, library/tool versions, per-episode
outcomes, decoded frame counts, and measured preparation time.

`pixel_frame_work` sums decoded frames multiplied by prepared width and height.
It is an input-size proxy. It is not processor visual tokens, model FLOPs, GPU
cost or inference latency; smaller input does not establish faster inference.
Preparation timing is a single CPU pass, includes tool startup, and excludes
source synthesis and the vision function. It is recorded for diagnosis and is
not the ranking criterion.

For an actual VLM experiment, author a new fixed evaluator using public or
authorized source episodes and your model callback. Retain episode/person/source
group boundaries when splitting, verify what the processor actually consumes,
and measure quality and end-to-end inference cost on matched inputs. Freeze the
candidate before opening independent confirmation. The eight-episode synthetic
criterion and 100% threshold are example choices, not general release policy.

## Check the example

```bash
uv run --locked --project examples/video_preparation_autoresearch \
  pytest -q examples/video_preparation_autoresearch/tests
uv run --locked --project examples/video_preparation_autoresearch \
  ruff check --fix examples/video_preparation_autoresearch
uv run --locked --project examples/video_preparation_autoresearch \
  ruff format examples/video_preparation_autoresearch
uv run --locked --project examples/video_preparation_autoresearch \
  ty check --project examples/video_preparation_autoresearch --extra-search-path . \
  examples/video_preparation_autoresearch
```

The tests use real synthetic videos and installed FFmpeg/ffprobe. They verify
the example's denominators, selected input budget, disjoint confirmation and
evidence-change rejection. HFlow core already tests its encoder and sampling
contracts; this example does not replace those tests.
