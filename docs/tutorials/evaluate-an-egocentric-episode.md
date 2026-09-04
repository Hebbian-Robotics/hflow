# Evaluate your first egocentric episode

This tutorial runs one real recording through HFlow's recommended starting
point for egocentric video: the default deterministic quality checks plus two
hosted model checks. It demonstrates the useful split in one command:

- HFlow measures recording integrity locally and deterministically.
- HFlow's hosted checks add semantic evidence without requiring an account,
  API key, or model server.
- Both kinds of evidence are versioned and recorded through the same pipeline.

The hosted checks are registered explicitly rather than included in
`hflow.checks.DEFAULT_CHECKS`. Constructing a normal `hflow.App` must remain
offline and must not send recording data over the network unexpectedly. This
tutorial makes that boundary visible in both the command and the source.

## 1. Install the repository environment

From the repository root:

```bash
uv sync --locked
uv tool install -U huggingface_hub
```

## 2. Download a real sample recording

The sample is a short first-person "Thimble Removal" episode from Lightwheel's
[`EgoDemo`](https://huggingface.co/datasets/LightwheelAI/EgoDemo) dataset. The
Hugging Face CLI returns its cache path; copy it to the short path used below:

```bash
mkdir -p data/episode-evaluation
downloaded_sample_mcap_path="$(
    hf download LightwheelAI/EgoDemo \
        'EgoStand/mcap/Thimble Removal/a8d29fea-3cf9-47f7-ad4b-4b1d0ecb7a71.mcap' \
        --repo-type dataset \
        --quiet
)"
cp "$downloaded_sample_mcap_path" data/episode-evaluation/sample.mcap
```

If Hugging Face requests access, accept the dataset's terms and run
`hf auth login`, then repeat the download.

## 3. Run the evaluation

The sample has two camera streams, so select its left head camera explicitly:

```bash
uv run python examples/evaluate_episode.py \
    data/episode-evaluation/sample.mcap \
    --camera /sensor/camera/head_left/video
```

No model endpoint or API key is required. HFlow processes the episode locally;
the two model checks send only the selected JPEG frame to `api.hflow.dev`.

The first sample frame visibly contains two wearer hands manipulating material.
The hosted results should therefore include:

```text
build_ai/hand_count/prediction = 2
build_ai/active_manipulation/prediction = yes
```

The report also includes HFlow's automatic deterministic baseline:

| Check | Evidence it records |
|---|---|
| `episode_duration` | Recording duration and message coverage |
| `timestamp_regularity` | Timing intervals, gaps, and cross-stream alignment |
| `camera_frame_stats` | Frame count, blackout, freeze, and exposure measurements |
| `keyframe_interval` | Video seekability evidence |
| `content_digest` | Exact recording identity for duplicate detection |
| `media_digest` | Visual-media identity across differently wrapped recordings |
| `build_ai_hand_visibility` | Number of wearer hands visible in the selected frame |
| `build_ai_active_manipulation` | Whether the wearer is visibly manipulating an object |

The canonical MCAP, contact sheets, run evidence, and Parquet catalog are
written beneath `data/episode-evaluation/`.

## 4. Evaluate your recording

Replace the sample path with your MCAP. Omit `--camera` when it contains only
one camera, or pass the desired camera topic when it contains several:

```bash
uv run python examples/evaluate_episode.py path/to/episode.mcap \
    --camera /your/egocentric/camera
```

Use `--frame-time-seconds` to choose another instant. The selected time and
camera are part of each model check's version, so results from different
sampling configurations do not silently claim to be comparable.

The hosted checks implement Build AI's published hand-visibility and
active-manipulation prompts. For custom prompts, models, or OpenAI-compatible
endpoints, continue with the
[Build AI evaluation guide](../how-to/run-build-ai-evaluation.md). To configure
or add deterministic checks, see the
[built-in checks guide](../how-to/enable-built-in-checks.md).

## Bring us your workflow

This starter pipeline is intentionally general. Real deployments usually need
checks and processing tailored to the recording rig, task, failure modes, and
training format—and a plan for running them across the full corpus.

If you want help designing that pipeline or evaluating what HFlow can automate
for your team, [tell Hebbian Robotics about your workflow](https://forms.gle/EZpQpGGF3eJomx498).
For implementation questions and open-source feedback, join the
[HFlow Discord](https://discord.gg/vacepQvjmg).
