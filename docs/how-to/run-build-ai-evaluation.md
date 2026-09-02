# Compare vision models on the Build AI evaluations

Apply Build AI's single-frame judgments as HFlow checks, or use their
Egocentric-10K and Egocentric-100K evaluation inputs to compare vision models
and prompt variants through any endpoint that implements OpenAI-compatible
Chat Completions with image inputs.

The example has two execution paths built around the same judgment contract:

- [`pipeline.py`](../../examples/build_ai_evaluation/pipeline.py) registers
  `build_ai_hand_visibility` and `build_ai_active_manipulation` as `hflow.App`
  checks for MCAP episodes.
  The checks return episode measurements plus timestamped model-output
  observations.
- [`evaluate.py`](../../examples/build_ai_evaluation/evaluate.py) streams the
  published Parquet frames into Inspect AI without transcoding them. Inspect
  handles OpenAI-compatible model execution, retries, structured logs, scoring,
  and the local results viewer; the adapter adds the pinned datasets and
  Build AI-specific summary and comparison table.

## Run the methodology on an HFlow episode

Configure any OpenAI-compatible vision endpoint and the environment variable
that contains its credential:

```bash
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
export OPENAI_MODEL="qwen/qwen3-vl-32b-instruct"
export BUILD_AI_API_KEY_ENV="OPENROUTER_API_KEY"
```

Run against a small synthesized episode:

```bash
uv run --project examples/build_ai_evaluation \
    python -m examples.build_ai_evaluation.pipeline
```

Or pass an existing MCAP recording:

```bash
uv run --project examples/build_ai_evaluation \
    python -m examples.build_ai_evaluation.pipeline path/to/episode.mcap
```

The pipeline samples one frame at zero seconds from the episode's only camera
and runs both judgments as `@app.check` steps. For a multi-camera episode, set
`BUILD_AI_CAMERA` to the camera topic. Change the sample time with
`BUILD_AI_FRAME_TIME_SECONDS`. The resulting HFlow measurements use the
`build_ai/hand_count/` and `build_ai/active_manipulation/` namespaces and
include the parsed prediction, raw response, requested and routed models, and
numeric token usage.

Each check also writes one `hflow.Observation` at the exact sampled-frame
timestamp. Its typed fields retain the task, prediction, raw response,
validity, requested/routed models, and usage in the catalog's `observations`
table. The measurements remain as convenient episode summaries; the
observation is the per-frame record that scales to evaluations sampling more
than one frame.

Swap prompt files with `BUILD_AI_HAND_VISIBILITY_PROMPT` and
`BUILD_AI_ACTIVE_MANIPULATION_PROMPT`. Other request controls are
`BUILD_AI_RESPONSE_FORMAT`, `BUILD_AI_TEMPERATURE`, `BUILD_AI_MAX_TOKENS`, and
`BUILD_AI_MAX_RETRIES`. The registered HFlow check version hashes every
result-affecting model, prompt, schema, request, camera, and sampling setting so
different configurations do not claim comparable step versions.

For an unauthenticated self-hosted endpoint, omit `BUILD_AI_API_KEY_ENV`. Each
check can use a different service through the
`BUILD_AI_HAND_VISIBILITY_BASE_URL` / `BUILD_AI_HAND_VISIBILITY_MODEL` /
`BUILD_AI_HAND_VISIBILITY_API_KEY_ENV` and corresponding
`BUILD_AI_ACTIVE_MANIPULATION_*` overrides.

The same checks are available directly in any pipeline:

```python
hflow.build_ai_vlm_checks.register_hand_visibility(
    app,
    endpoint="http://localhost:8000/v1",
    model="Qwen/Qwen3-VL-8B-Instruct",
)
hflow.build_ai_vlm_checks.register_active_manipulation(
    app,
    endpoint="https://hosted.example/v1",
    model="hosted-model",
    api_key_environment_variable="HOSTED_MODEL_API_KEY",
)
```

The API key is optional for local unauthenticated servers. Registration is
opt-in rather than part of `DEFAULT_CHECKS`, because both checks perform model
calls and users may intentionally choose different models for them.

## What this reproduces

The original specification is Build AI's
[Egocentric-10K evaluation release](https://huggingface.co/datasets/builddotai/Egocentric-10K-Evaluation/tree/d74b7883c998dd360e3f051830fcc792a83985e6),
including its exact
[hand-visibility prompt](https://huggingface.co/datasets/builddotai/Egocentric-10K-Evaluation/blob/d74b7883c998dd360e3f051830fcc792a83985e6/prompts/hand_count.txt)
and
[active-manipulation prompt](https://huggingface.co/datasets/builddotai/Egocentric-10K-Evaluation/blob/d74b7883c998dd360e3f051830fcc792a83985e6/prompts/active_manipulation.txt),
plus the subsequent
[Egocentric-100K evaluation release](https://huggingface.co/datasets/builddotai/Egocentric-100K-Evaluation/tree/d0f69a56b0525c1bead80d918dc57ef83dcac899).
Those two evaluations use the same disclosed method:

1. Randomly select 10,000 frames from each of the Build AI, Ego4D, and
   EPIC-KITCHENS corpora.
2. Ask one hand-count question and one active-manipulation question about each
   individual frame.
3. Use a structured response schema and report label prevalence.

The published evaluation Parquet files preserve those selected frames and the
Gemini 2.5 Flash labels. The runner pins the evaluation repositories to
immutable revisions, reuses those exact frames, and defaults to the upstream
prompts and response schemas exposed by `hflow.build_ai_vlm_checks`. Prompt files
passed to the CLI override those defaults for comparison runs.

The 10K files include upstream UUID frame IDs. The 100K files do not, so the
runner assigns deterministic `row-00000`-style IDs in pinned Parquet order for
Inspect's per-sample logs; this does not change the images, selection, or
evaluation order.
The Egocentric-100K file also encodes its active-manipulation labels as
`true`/`false`; the runner normalizes those published values to the same
`yes`/`no` result vocabulary used by the other five files.
Its internal source column contains two collection-shard names; summaries use
the enclosing published corpus name so those shards remain one Egocentric-100K
headline row.

Each full version contains 30,000 frames: 10,000 from the named Build AI corpus,
10,000 from Ego4D, and 10,000 from EPIC-KITCHENS. Two requests per frame means
60,000 model requests. `--source build` runs only the 10,000 Build AI frames
(20,000 requests), while `--limit N` runs a deterministic first-`N` subset per
selected corpus for development and cost checks.

This is a prevalence evaluation used to benchmark models, backed by
model-generated reference labels rather than human ground truth. Agreement
says how often a run matches Build AI's published Gemini labels; it does not
establish that either label is correct. Build AI disclosed the model but not every
generation, provider-routing, or random-sampling setting. Replaying the pinned
frames removes selection ambiguity, but a current Gemini endpoint can still
differ from the endpoint used for the original publication.

## Prerequisites and side effects

Install the Build AI workspace project's locked environment, then install the
current Hugging Face CLI:

```bash
uv sync --locked --project examples/build_ai_evaluation
uv tool install -U huggingface_hub
```

The example downloads public Parquet files from Hugging Face when
`--download` is present. All three corpora need approximately 5.5 GB for the
10K evaluation or 6 GB for the 100K evaluation; the Build-only files are
approximately 1.8 GB and 2.3 GB respectively. Downloads and run artifacts land
under the gitignored `data/build-ai-evaluation/` directory by default.

Every evaluated task makes an external model request. A full three-corpus run
makes 60,000 requests and can incur substantial API charges. Start with
`--source build --limit 1`, inspect the result, then increase the limit
deliberately. Credentials are read from the environment variable named by
`--api-key-env`; they are never copied into run metadata or Inspect logs.

## Replay the published evaluation with Gemini 2.5 Flash through OpenRouter

OpenRouter currently exposes Gemini 2.5 Flash as
[`google/gemini-2.5-flash`](https://openrouter.ai/google/gemini-2.5-flash/api),
accepts base64 image data through its OpenAI-compatible Chat Completions API,
and supports JSON Schema structured outputs for compatible providers.

Keep the OpenRouter key in its existing environment variable:

```bash
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
export OPENAI_MODEL="google/gemini-2.5-flash"
```

Run a one-frame, two-request smoke evaluation first:

```bash
uv run --project examples/build_ai_evaluation \
    python examples/build_ai_evaluation/evaluate.py run \
    --dataset 10k \
    --source build \
    --limit 1 \
    --download \
    --api-key-env OPENROUTER_API_KEY \
    --label gemini-2.5-flash-smoke \
    --output data/build-ai-evaluation/runs/gemini-2.5-flash-smoke
```

Then reproduce both published three-corpus evaluation versions. Omitting
`--source` selects all three corpora and omitting `--limit` selects every row:

```bash
uv run --project examples/build_ai_evaluation \
    python examples/build_ai_evaluation/evaluate.py run \
    --dataset 10k \
    --download \
    --api-key-env OPENROUTER_API_KEY \
    --label gemini-2.5-flash-build-ai-10k \
    --output data/build-ai-evaluation/runs/gemini-2.5-flash-build-ai-10k

uv run --project examples/build_ai_evaluation \
    python examples/build_ai_evaluation/evaluate.py run \
    --dataset 100k \
    --download \
    --api-key-env OPENROUTER_API_KEY \
    --label gemini-2.5-flash-build-ai-100k \
    --output data/build-ai-evaluation/runs/gemini-2.5-flash-build-ai-100k
```

The default response mode is `json-schema`, matching the published structured
method. Structured-output support can vary by provider endpoint. Use
`--response-format json-object` or `--response-format text` for a compatible
server that does not implement JSON Schema; the result parser accepts both the
published JSON objects and plain `0`/`1`/`2` or `yes`/`no` answers.

## Swap the model or prompts

Change only the model to compare models under the same frame, prompt, and
request contract:

```bash
uv run --project examples/build_ai_evaluation \
    python examples/build_ai_evaluation/evaluate.py run \
    --dataset 10k \
    --source build \
    --model your-provider/your-vision-model \
    --base-url https://your-endpoint.example/v1 \
    --api-key-env YOUR_ENDPOINT_API_KEY \
    --label your-vision-model \
    --output data/build-ai-evaluation/runs/your-vision-model-10k
```

Change either prompt with a tracked text file. Use a new output directory for
every result-changing configuration:

```bash
uv run --project examples/build_ai_evaluation \
    python examples/build_ai_evaluation/evaluate.py run \
    --dataset 10k \
    --source build \
    --hand-count-prompt path/to/hand-count-v2.txt \
    --active-manipulation-prompt path/to/active-manipulation-v2.txt \
    --model your-provider/your-vision-model \
    --base-url https://your-endpoint.example/v1 \
    --api-key-env YOUR_ENDPOINT_API_KEY \
    --label prompt-v2 \
    --output data/build-ai-evaluation/runs/prompt-v2-10k
```

For an unauthenticated self-hosted endpoint, add
`--allow-missing-api-key`. `--temperature` is intentionally omitted by
default because Build AI did not publish one; set it explicitly when a model
comparison requires a fixed non-default value. The run fingerprint captures
the endpoint, model, prompt contents, schema mode, temperature, token limit,
selected rows, sources, and tasks so incompatible experiments cannot silently
reuse the same output directory.

## Inspect and compare results

Each output directory contains:

| File | Contents |
| --- | --- |
| `run.json` | Pinned dataset revision, model and endpoint, full prompt text and hashes, request settings, and run fingerprint |
| `logs/*.eval` | Inspect logs with each input, raw model output, normalized score, reference target, errors, latency, and token usage |
| `summary.json` | Per-corpus prevalence, valid/invalid/error counts, agreement, latency, and aggregate usage |

Inspect retries retryable endpoint failures up to `--max-retries`, limits live
requests with `--workers`, and writes sample logs as the run progresses. A
repeated command is a new repeat of the same experiment and adds another set of
Inspect logs. A changed result-affecting setting fails with an instruction to
choose another output directory.

Open the Inspect viewer for a run directory:

```bash
uv run --project examples/build_ai_evaluation inspect view \
    --log-dir data/build-ai-evaluation/runs/gemini-2.5-flash-build-ai-10k/logs
```

Compare any completed summaries without another endpoint call:

```bash
uv run --project examples/build_ai_evaluation \
    python examples/build_ai_evaluation/evaluate.py compare \
    data/build-ai-evaluation/runs/gemini-2.5-flash-build-ai-10k/summary.json \
    data/build-ai-evaluation/runs/your-vision-model-10k/summary.json
```

The printed table matches Build AI's headline prevalence columns and adds
agreement with the published labels. Read every percentage next to its valid
`n`: invalid or failed responses are excluded from prevalence rather than
silently counted as negative labels.

Sources: the pinned
[Egocentric-10K evaluation](https://huggingface.co/datasets/builddotai/Egocentric-10K-Evaluation/tree/d74b7883c998dd360e3f051830fcc792a83985e6),
[Egocentric-100K evaluation](https://huggingface.co/datasets/builddotai/Egocentric-100K-Evaluation/tree/d0f69a56b0525c1bead80d918dc57ef83dcac899),
and OpenRouter's
[structured-output contract](https://openrouter.ai/docs/guides/features/structured-outputs).
