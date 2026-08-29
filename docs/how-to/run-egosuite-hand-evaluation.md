# Evaluate VLM hand counts against EgoSuite joint labels

Use Lightwheel EgoSuite's synchronized hand pose and camera calibration as an
independent geometric reference, then compare that reference with the hand
count produced by any vision model available through an OpenAI-compatible Chat
Completions endpoint.

The runnable example is
[`examples/egosuite_evaluation/evaluate.py`](../../examples/egosuite_evaluation/evaluate.py).
It has two commands:

- `labels` calculates the projected labels without an API key or model call.
- `run` extracts the same selected images and asks a VLM to count the visible
  camera wearer's hands through Inspect AI.

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
labels and ffmpeg image extraction use those exact source-frame indices.

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
the exact source frame index, and any pose-quality reasons. Run `labels` first
on a new corpus to inspect class balance before interpreting model accuracy.

For a class-balanced comparison, add `--samples-per-hand-count N`. The command
deterministically selects up to `N` frames from each projected class across all
input episodes. `--sample-seed` defaults to 42, and both values are recorded in
the label report or run contract. `--frame-stride` is applied first, so use
`--frame-stride 1` when every source frame should be eligible for sampling.

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

After inspecting the full label distribution, run a declared stratified sample
over a directory or several MCAP paths:

```bash
uv run --project examples/egosuite_evaluation \
    python examples/egosuite_evaluation/evaluate.py run \
    data/egosuite-evaluation/datasets/EgoDemo/EgoStand/mcap \
    --frame-stride 1 \
    --samples-per-hand-count 100 \
    --sample-seed 42 \
    --api-key-env OPENROUTER_API_KEY \
    --output data/egosuite-evaluation/runs/gemma-stratified-300
```

This requests at most 300 images. If a class has fewer than 100 eligible
frames, all of its frames are retained and the printed target counts expose the
shortfall.

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

Agreement must be interpreted with the reference distribution and confusion
matrix. If nearly every selected frame contains two projected hands, a model
that always answers `2` can have high accuracy without solving the minority
cases. Expand the episode set or use `--samples-per-hand-count` before making a
comparative capability claim.
