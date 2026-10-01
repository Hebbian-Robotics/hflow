# Build AI tiny-VLM autoresearch

Prepare public data, measure a tiny VLM, fine-tune a LoRA adapter, and let a
coding agent improve three recipe values under a fixed budget. It builds on the existing
[Build AI evaluation](../../docs/how-to/run-build-ai-evaluation.md), using its
published teacher-labelled frames and HFlow's general manifest splitter.

## Prepare both releases from scratch

Prerequisites: Python 3.11+, uv, local disk for approximately 12 GB of source
Parquet plus retained image files. Model weights, GPU, Gemini credentials, and
model API calls are not required for this stage. Install the example environment:

```bash
uv sync --locked --project examples/build_ai_autoresearch
```

Explicitly permit downloads and prepare a fresh output directory:

```bash
uv run --locked --project examples/build_ai_autoresearch \
  python examples/build_ai_autoresearch/prepare.py \
  data/build-ai-autoresearch/prepared --download
```

Downloads are pinned to the same immutable releases used by the existing
evaluation adapter:

- [Egocentric-10K evaluation](https://huggingface.co/datasets/builddotai/Egocentric-10K-Evaluation/tree/d74b7883c998dd360e3f051830fcc792a83985e6)
- [Egocentric-100K evaluation](https://huggingface.co/datasets/builddotai/Egocentric-100K-Evaluation/tree/d0f69a56b0525c1bead80d918dc57ef83dcac899)

Each release contains 10,000 frames per corpus (Build AI, Ego4D, EPIC-KITCHENS):
60,000 input rows before deduplication. The task in this first version is
wearer-visible hand count: 0, 1, or 2. The references are published
Gemini-generated teacher labels, not human ground truth. Consult the dataset
cards and source-corpus terms when using the distributed material.

For a small CPU pilot, add `--limit 20`. This limits selected rows per file,
but downloads still fetch full Parquet files. A first-row subset is a pipeline
check, not a representative quality benchmark. Existing local files can be
used without downloads:

```bash
uv run --locked --project examples/build_ai_autoresearch \
  python examples/build_ai_autoresearch/prepare.py \
  data/build-ai-autoresearch/local-pilot \
  --cache data/build-ai-evaluation/datasets --limit 20
```

Both downloaded and locally cached files must match the published Git LFS
SHA-256 values at the pinned revisions. Arbitrary files placed in a cache
directory are rejected. Preparation records download/local origin separately
from content verification, then checks source digests again after processing;
changing inputs aborts the operation. Programmatic synthetic fixtures may
omit an expected digest and are explicitly marked revision-unverified.

## Mixing, deduplication, and splits

1. Read all six files in a fixed order, in small image batches.
2. Hash encoded bytes, then EXIF-oriented RGB dimensions and pixels. Identical
   encoded images reuse their decoded identity. Different encodings with the
   same pixels collapse into one sample.
3. Keep every original release/corpus/row/frame ID, encoded digest, and label
   in the sample's provenance. UUID frame IDs are not recording identities.
4. Exclude images whose copies have conflicting hand-count labels. Record
   all conflicting references; do not choose a majority label.
5. Call `hflow.split_manifest` on the deduplicated manifest to freeze
   train/development/test partitions with seed 42 and fractions 0.8/0.1/0.1.

**These are exact-pixel frame-level splits.** The published adapter does not
provide reliable recording identities. Pixel deduplication cannot identify
near duplicates, neighbouring frames, shared workers, or recording overlap.
Confirmation here measures agreement on unseen exact frames, not independence
of recordings or human-labelled correctness. Stronger evaluation requires
recording/participant identities and an independent human-labelled set.

## Outputs

The output directory must be new. It contains:

- `images/`: original encoded bytes of one representative per retained image.
  `.image` names avoid guessing the original format; Pillow reads the header.
- `samples.parquet`: sample identity, decoded pixel digest, image path relative
  to the prepared dataset root, encoded digest, dimensions, teacher label, and
  all source references as JSON.
- `conflicts.json`: contradictory hand labels and their provenance.
- `splits/`: partition Parquet files, assignments, and HFlow's split receipt.
- `preparation.json`: source digests, origin verification scope, selection
  limits, counts, deduplication policy, code/runtime identities, output hashes,
  and split receipt hash.

Receipts are written last. Ordinary failures remove the new output directory;
termination can leave incomplete output. Consumers must verify the preparation
receipt, referenced file digests, and image digests before a model run.
Images should be EXIF-transposed and converted to RGB for training/evaluation,
matching preparation's deduplication semantics. No resizing happens here.
All generated data belongs under ignored `data/`; do not commit frames.

## Run a CPU reference experiment

Training is an optional example dependency stack. Its PyTorch and torchvision
wheels come from the official CPU index; no GPU, paid model API, or Gemini key
is needed. The first model run downloads public weights/tokenizer/processor
files. Use Python 3.11+ on Linux x86-64 for the validated setup and allow several
GB of RAM and disk. CUDA serving/training is not validated by this example.

```bash
uv sync --locked --project examples/build_ai_autoresearch --extra training
```

The reference model is
[SmolVLM2-256M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct/tree/067788b187b95ebe7b2e040b3e4299e342e5b8fd),
pinned to an immutable revision. Although video-capable, it receives one image
per example. We reuse HFlow's public wearer-hand prompt, EXIF-oriented RGB,
512-pixel processing with image splitting disabled, greedy decoding, and at
most four output tokens. Only a stripped `0`, `1`, or `2` is a valid answer;
invalid answers stay in the metric denominator. This is an explicitly named
Transformers CPU reference backend, with a required recorded reason.

Freeze a small protocol. The reason is a positional argument:

```bash
uv run --locked --project examples/build_ai_autoresearch --extra training \
  python -m examples.build_ai_autoresearch.experiment init \
  data/build-ai-autoresearch/prepared data/build-ai-autoresearch/experiment \
  'CPU reference pilot before allocating GPU compute' \
  --train-samples 192 --development-samples 48 --confirmation-samples 48 \
  --training-steps 256 --max-trials 2
```

The defaults allow eight trials, 256 optimizer steps each, 192 training frames,
and 48 frames each for development and confirmation. Selection is deterministic
for seed 42. Training reserves one frame per class, then follows the original
sample pool. Development/confirmation choose classes in round-robin order;
their scores describe these balanced subsets, not the original class prior.
Small preparation pilots may lack a class; initialization rejects those.

Run the untouched baseline, then the naive recipe already copied into the
experiment directory:

```bash
uv run --locked --project examples/build_ai_autoresearch --extra training \
  python -m examples.build_ai_autoresearch.experiment baseline-transformers-reference \
  data/build-ai-autoresearch/experiment
uv run --locked --project examples/build_ai_autoresearch --extra training \
  python -m examples.build_ai_autoresearch.experiment trial-transformers-reference \
  data/build-ai-autoresearch/experiment
```

Each trial starts from the same base weights. LoRA targets only language-model
query/value projections, with alpha twice the rank, no dropout, AdamW, batch
size one, and gradient clipping at 1. Loss covers assistant answer/end tokens;
the exact tokenized prompt prefix and padding are masked. The naive recipe
uses learning rate 0.0002, rank 4, and uniform training sampling.

## Agent search and frozen confirmation

Give your coding agent [program.md](./program.md), the experiment path, and the
trial command above. It edits **only `<experiment>/candidate.json`**:

```json
{"learning_rate": 0.0002, "lora_rank": 4, "sampling_balance": "uniform"}
```

Allowed ranks are 2, 4, and 8; sampling is `uniform` or `class-balanced`;
learning rate must be positive and at most 0.01. Extra fields are rejected.
The agent chooses the next recipe from development reports and records its
hypothesis. This is the autoresearch loop; no model API or particular coding
agent is built into the runner. First establish measurable headroom with the
baseline and naive trial before spending the remaining budget.

Trials report fixed three-class macro-F1, accuracy, invalid outputs, per-class
support/F1, and per-corpus agreement. A cross-corpus duplicate appears in each
source-corpus breakdown, but only once overall. Selection uses strict macro-F1
improvement; ties keep the earlier selection, including the untouched baseline.
Failed/interrupted attempts consume the trial budget. Run commands serially.

The operator then ends search and confirms once:

```bash
uv run --locked --project examples/build_ai_autoresearch --extra training \
  python -m examples.build_ai_autoresearch.experiment freeze \
  data/build-ai-autoresearch/experiment
uv run --locked --project examples/build_ai_autoresearch --extra training \
  python -m examples.build_ai_autoresearch.experiment confirm-transformers-reference \
  data/build-ai-autoresearch/experiment data/build-ai-autoresearch/prepared
```

Initialization copies only training/development images. It records the test
manifest digest without opening that partition. **The runner is not an agent
sandbox.** Keep the prepared dataset, original source files, and confirmation
command inaccessible to the editing agent, for example in a separate operator
account/container. Receipt checks detect drift; filesystem permissions enforce
access. Keep evaluator code, the dependency lock, and protocol read-only.

Selection is written exclusively and blocks further trials. Confirmation
creates an exclusive directory before loading the model; a failed attempt
also consumes confirmation. It materializes test images there, so keep that
directory away from the search agent. Do not restart search on the same test
set after viewing confirmation. A failed or disappointing result is a result.

Protocol/evaluator/runtime identities, sample hashes, recipes, predictions,
losses, and adapter-file hashes are recorded in JSON receipts. Changed code,
lockfiles, images, or selected checkpoints are rejected. Interrupted directories
without their final receipt are incomplete; use a new experiment for development
debugging without treating an exposed confirmation set as fresh evidence.

## Use the result with HFlow

Export the confirmed selection as a standard model/processor directory. If no
trial beat the baseline, the exported model is the untouched reference model:

```bash
uv run --locked --project examples/build_ai_autoresearch --extra training \
  python -m examples.build_ai_autoresearch.experiment export-transformers-reference \
  data/build-ai-autoresearch/experiment data/build-ai-autoresearch/export
```

The output includes `model/` (merged LoRA weights when selected) and a receipt
binding its files to selection and confirmation. Serve that directory through
your separately provisioned OpenAI-compatible VLM endpoint, then follow the
existing [HFlow Build AI evaluation guide](../../docs/how-to/run-build-ai-evaluation.md).
Endpoint/model support must be checked for your serving stack; the CPU pilot
does not validate vLLM deployment. That published evaluation pool overlaps the
training pool here, so a rerun on it is an integration check, not independent
quality evidence.

This example contributes the reusable experiment workflow. Teacher agreement,
short training runs, and exact-frame separation do not establish production
readiness or recording-independent generalization.

## CPU pilot evidence

A local four-thread CPU development pilot used 192 training frames, 48 balanced
development frames, 256 steps, learning rate 0.0002, rank 8, and class-balanced
sampling. Baseline macro-F1 was 0.1667; the adapter reached 0.4889 (29/48 teacher
agreements, no invalid answers). Training took approximately 268 seconds and
development evaluation 38 seconds on that host. Class F1 was 0.6667/0/0.8:
the one-hand class still failed. This establishes limited development headroom,
not a useful production model or fresh confirmation evidence.

A separate 32-step pilot stayed at constant answers. Its frozen baseline
fallback, one-time confirmation, and export completed. Adapter reload preserved
all development predictions; adapter merge/export also worked. After the final
runner changes, a tiny two-step run repeated baseline/trial/selection/
confirmation/export as a control-flow check. These smoke checks establish
execution, not quality. Generated frames, predictions, and weights remain under
ignored `data/`; no data or model artifacts are distributed in this example.

Validation:

```bash
uv run --locked --project examples/build_ai_autoresearch \
  pytest -q examples/build_ai_autoresearch/tests
uv run --locked --project examples/build_ai_autoresearch --extra training \
  ruff check --fix examples/build_ai_autoresearch
uv run --locked --project examples/build_ai_autoresearch --extra training \
  ruff format examples/build_ai_autoresearch
uv run --locked --project examples/build_ai_autoresearch --extra training \
  ty check --project examples/build_ai_autoresearch examples/build_ai_autoresearch
```

The outcome tests use synthetic prepared images and receipt fixtures; they do
not download models or pretend to establish fine-tuning quality. Validate a
real baseline/trial/confirmation cycle separately as above.
