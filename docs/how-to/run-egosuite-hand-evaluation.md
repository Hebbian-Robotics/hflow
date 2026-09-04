# Evaluate VLM hand counts against EgoSuite joint labels

Use Lightwheel EgoSuite's synchronized hand pose and camera calibration as an
independent geometric reference, then compare that reference with the hand
count produced by any vision model available through an OpenAI-compatible Chat
Completions endpoint.

The runnable example is
[`examples/egosuite_evaluation/evaluate.py`](../../examples/egosuite_evaluation/evaluate.py).
It has two commands:

- `labels` uses HFlow's multi-topic decoded batches to calculate the projected
  labels without an API key or model call.
- `run` uses HFlow's exact source-index frame extraction and asks a VLM to
  count the visible camera wearer's hands through Inspect AI.

## Reference-label contract

Each annotated EgoSuite MCAP provides frame-aligned messages for:

- 21 left-hand and 21 right-hand joints in world coordinates;
- a head-camera child-frame pose in the world parent frame;
- camera width, height, and the 3×3 intrinsic matrix;
- head-camera H.264 video; and
- per-hand pose-quality issues such as `occlusion` and `near edge`.

For each selected source frame, the example:

1. Inverts the world-parent to camera-child pose to express every joint in
   camera coordinates.
2. Rejects joints at non-positive camera depth.
3. Applies the pinhole intrinsic matrix.
4. Marks a hand in frame when at least one projected joint lies within
   `0 <= x < width` and `0 <= y < height`.
5. Adds the two per-hand booleans to produce a reference count of 0, 1, or 2.

The default `--frame-stride 30` selects frames 0, 30, 60, and so on, which is
about one frame per second for the 30 fps EgoDemo recordings. Both the joint
labels and `hflow.Episode.frames_at_indices()` use those exact source-frame
indices.

This is a ground-truth label for **projected hand-joint presence**, not a pixel
segmentation label. It establishes that labeled hand geometry lies in the
camera frustum. It cannot prove that the pixels are unobstructed, and a tiny
visible hand edge with no labeled joint inside the image can be missed. The
source pose-quality flags are therefore preserved in every sample and summary
as secondary audit metadata; they do not turn a projected hand into an absent
hand because an `occlusion` can still leave part of the hand visible.

The runner rejects nonzero lens-distortion coefficients instead of silently
applying an incomplete projection. EgoDemo's inspected head-camera images are
rectified and satisfy this contract.

## Why this example does not score active manipulation

EgoSuite's hand joints support the geometric hand-presence proxy above, but
they do not label object contact, grasp state, or active manipulation. Joint
speed is not an equivalent target: a person can actively hold or position an
object without moving quickly, and ordinary arm or camera motion can produce a
high speed without object manipulation.

The semantic segments are also too coarse to repair that gap. We audited
candidate negative phases from the pinned data before adding such a score:

- low-motion shelf-label frames still show the wearer deliberately holding a
  sign in position;
- `Examine and compare the toothbrushes on the shelf` includes both hands
  pulling and comparing packaged toothbrushes; and
- `Close the lid of the device and wait for the paper to come out` continues
  to show both hands on the label printer during the apparent wait.

All three can reasonably be `yes` under the Build AI single-image definition,
even when a motion or phase-name heuristic says `no`. Treating those proxies as
ground truth would measure the heuristic's errors as model errors. A defensible
active-manipulation extension therefore needs frame-level human labels or an
additional contact/grasp annotation source. The current example intentionally
limits its reference claim to projected hand-joint presence.

## Prerequisites and side effects

Install the workspace example and the current Hugging Face CLI:

```bash
uv sync --locked --project examples/egosuite_evaluation
uv tool install -U huggingface_hub
```

You also need ffmpeg and ffprobe on `PATH`. Accept the access terms for
[`LightwheelAI/EgoDemo`](https://huggingface.co/datasets/LightwheelAI/EgoDemo)
and authenticate with `hf auth login` before downloading. EgoSuite uses the
`commercial-training-no-resale-v1.0` dataset license; review its terms for your
use case.

The example writes downloads and generated artifacts under
`data/egosuite-evaluation/`, which is gitignored. A model run sends the selected
JPEGs to the endpoint configured by `--base-url` or `OPENAI_BASE_URL` and may
incur provider charges. A self-hosted endpoint avoids third-party image egress
when its own deployment and logging are under your control.

## Download a small pinned episode

EgoDemo is a 50-hour entry point containing all four annotated EgoSuite
subsets in both LeRobot v3 and MCAP. This 9.2-second EgoStand episode is about
15 MB and is enough for the first run:

```bash
hf download LightwheelAI/EgoDemo \
    --repo-type dataset \
    --revision 08fe71c14b4a9d7dd891e729788be034b4b6bbb1 \
    --include 'EgoStand/mcap/Thimble Removal/a8d29fea-3cf9-47f7-ad4b-4b1d0ecb7a71.mcap' \
    --local-dir data/egosuite-evaluation/datasets/EgoDemo
```

Set a short path for the commands below:

```bash
export EGOSUITE_SAMPLE_MCAP='data/egosuite-evaluation/datasets/EgoDemo/EgoStand/mcap/Thimble Removal/a8d29fea-3cf9-47f7-ad4b-4b1d0ecb7a71.mcap'
```

## Run the methodology as an HFlow check

The HFlow pipeline applies the projection and image-only model judgment to one
annotated episode. Configure any OpenAI-compatible endpoint and run:

```bash
export OPENAI_BASE_URL='https://openrouter.ai/api/v1'
export OPENAI_MODEL='google/gemma-4-26b-a4b-it'
export EGOSUITE_API_KEY_ENV='OPENROUTER_API_KEY'
uv run --project examples/egosuite_evaluation \
    python -m examples.egosuite_evaluation.pipeline "$EGOSUITE_SAMPLE_MCAP"
```

The `egosuite_projected_hand_visibility` check uses HFlow to decode the
synchronized annotation topics and extract exact source frames. It records
reference and predicted class counts, valid-output and agreement fractions,
token usage, and zero-length timestamp intervals for mismatches or unparsed
answers. The default is a cost-safe ten frames at `--frame-stride 30`
equivalent sampling. Configure it with `EGOSUITE_FRAME_STRIDE`,
`EGOSUITE_LIMIT_PER_EPISODE`, `EGOSUITE_CAMERA`, `EGOSUITE_RESPONSE_FORMAT`,
and the other `EGOSUITE_*` environment variables defined in
[`pipeline.py`](../../examples/egosuite_evaluation/pipeline.py).

To apply an exact saved slice rather than resampling each episode, point the
pipeline at a label report produced by the `labels` command:

```bash
export EGOSUITE_LABEL_MANIFEST='data/egosuite-evaluation/labels/natural-1000.json'
```

The pipeline validates every manifest record, selects its declared source
frame indices only when the report's `source_uri` matches the canonical
episode's HFlow provenance, and includes the manifest digest in the HFlow
check version. Missing or mismatched provenance fails before frame extraction
or a model request. Missing or malformed model content is retained as a
`valid=false` observation so provider output failures lower end-to-end
accuracy instead of discarding the episode's evidence.

Current label reports record the same source identity returned by
`App.source_identity()`. Set `HFLOW_DATA_ROOT` consistently when creating the
report and running the pipeline. When the dataset lives below that root, the
identity is root-relative and remains stable across equivalent host and
container mount paths. For the layout in this guide, use:

```bash
export HFLOW_DATA_ROOT='data/egosuite-evaluation'
```

Reports created before `source_uri` was added remain usable when each
`source_episode` basename refers to exactly one source path. An older report
that uses one basename for multiple paths is ambiguous and must be regenerated.

Use this entrypoint when the judgment belongs in an HFlow episode pipeline.
Use `evaluate.py` below when comparing models over a declared dataset slice;
Inspect retains each raw response and produces cross-episode evaluation
summaries.

## Calculate the projected labels

Calculate every source frame without making any network request:

```bash
uv run --project examples/egosuite_evaluation \
    python examples/egosuite_evaluation/evaluate.py labels \
    "$EGOSUITE_SAMPLE_MCAP" \
    --frame-stride 1 \
    --output data/egosuite-evaluation/labels/thimble-removal.json
```

The terminal table reports the number of frames labeled 0, 1, or 2. The JSON
file records the projected joint count for each hand, the resulting hand count,
the exact source frame index, HFlow's canonical `source_uri`, and any
pose-quality reasons. Run `labels` first on a new corpus to inspect class
balance before interpreting model accuracy.

For a naturally distributed random sample, select episodes and then select
temporally separated frames within each episode:

```bash
uv run --project examples/egosuite_evaluation \
    python examples/egosuite_evaluation/evaluate.py labels \
    data/egosuite-evaluation/datasets/EgoDemo \
    --episode-count 100 \
    --frame-stride 30 \
    --samples-per-episode 10 \
    --sample-seed 42 \
    --output data/egosuite-evaluation/labels/natural-1000.json
```

`--episode-count` samples from the MCAPs available under the supplied inputs;
it does not download missing episodes. Sampling the same number of frames from
each selected episode prevents long recordings from dominating. The frame
stride is applied before random selection, so the command above samples at
most ten one-per-second candidates from each of 100 episodes.

The natural sample is the appropriate primary result when measuring expected
performance on the source distribution. The summary also reports per-class
agreement and macro agreement so a frequent class cannot hide failures on a
rare class.

For a class-balanced diagnostic, add `--samples-per-hand-count N`. The command
deterministically selects up to `N` frames from each projected class after the
episode and frame selection. `--sample-seed` defaults to 42, and every
selection value is recorded in the label report or run contract. Use
`--frame-stride 1` when every source frame should be eligible for the
diagnostic slice.

Pass a directory to recursively include every MCAP under it:

```bash
uv run --project examples/egosuite_evaluation \
    python examples/egosuite_evaluation/evaluate.py labels \
    data/egosuite-evaluation/datasets/EgoDemo/EgoStand/mcap \
    --frame-stride 30
```

Use `--camera head-right` to evaluate the other stereo head image. Wrist
cameras are intentionally outside this hand-count example because the target
methodology concerns the wearer's first-person head view.

## Run a VLM through an OpenAI-compatible endpoint

Configure the endpoint and model. The variable named by `--api-key-env` can
have any name; it is not required to be an OpenAI credential:

```bash
export OPENAI_BASE_URL='https://openrouter.ai/api/v1'
export OPENAI_MODEL='google/gemma-4-26b-a4b-it'
```

Run ten regularly spaced frames as a cost-safe check:

```bash
uv run --project examples/egosuite_evaluation \
    python examples/egosuite_evaluation/evaluate.py run \
    "$EGOSUITE_SAMPLE_MCAP" \
    --frame-stride 30 \
    --limit-per-episode 10 \
    --api-key-env OPENROUTER_API_KEY \
    --output data/egosuite-evaluation/runs/gemma-thimble-removal
```

After inspecting the full label distribution, run a declared natural sample
over a directory or several MCAP paths:

```bash
uv run --project examples/egosuite_evaluation \
    python examples/egosuite_evaluation/evaluate.py run \
    data/egosuite-evaluation/datasets/EgoDemo/EgoStand/mcap \
    --episode-count 100 \
    --frame-stride 30 \
    --samples-per-episode 10 \
    --sample-seed 42 \
    --api-key-env OPENROUTER_API_KEY \
    --output data/egosuite-evaluation/runs/gemma-natural-1000
```

This requests at most 1,000 images. If an episode has fewer than ten eligible
frames, all of its frames are retained and the printed target counts expose the
shortfall. Add `--samples-per-hand-count` to create a separate rare-case
diagnostic run; do not describe its constructed class proportions as the
dataset's natural distribution.

For an unauthenticated self-hosted endpoint, add
`--allow-missing-api-key`. The model, endpoint, prompt, response format,
sampling parameters, source paths and sizes, camera, and projection contract
feed the run fingerprint. Reusing an output directory with a different
experiment fails rather than mixing results.

The output directory contains:

- `run.json`: complete run contract and fingerprint;
- `frames/`: cached JPEGs at the labeled source indices;
- `logs/`: Inspect AI per-sample model outputs and scores; and
- `summary.json`: validity counts, target and prediction distributions,
  valid-response and end-to-end accuracy, confusion matrix, latency, usage,
  and pose-quality metadata. End-to-end accuracy counts invalid or failed
  responses as incorrect instead of silently excluding them.

Swap the prompt with `--prompt path/to/prompt.txt`. Choose
`--response-format json-schema`, `json-object`, or `text`; leave
`--temperature` unset to use the provider default. The 512-token default leaves
room for endpoints that consume reasoning tokens before emitting the small JSON
answer; lower it only after confirming the selected model still returns valid
responses.

## Compare models

Give completed summaries readable labels with `--label`, then compare them:

```bash
uv run --project examples/egosuite_evaluation \
    python examples/egosuite_evaluation/evaluate.py compare \
    data/egosuite-evaluation/runs/gemma-thimble-removal/summary.json \
    data/egosuite-evaluation/runs/qwen-thimble-removal/summary.json
```

Agreement must be interpreted with the reference distribution, per-class
agreement, macro agreement, and confusion matrix. If nearly every selected
frame contains two projected hands, a model that always answers `2` can have
high natural-sample accuracy without solving the minority cases. Report that
distributional result as-is, then use a separately identified
`--samples-per-hand-count` slice to compare rare-case behavior.
