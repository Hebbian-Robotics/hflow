# Build AI tiny-VLM autoresearch

Prepare public data, measure a tiny VLM, fine-tune a LoRA adapter, and let a
coding agent improve one training file under a fixed compute budget. It builds on the existing
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
is needed. The first baseline run downloads public weights/tokenizer/processor
files. Use Python 3.11+ on Linux x86-64 for the validated setup and allow several
GB of RAM and disk. CUDA serving/training is not validated by this example.
Keep experiment outputs on a local filesystem supporting hard links, which
the receipt writer uses for exclusive atomic publication.

```bash
uv sync --locked --project examples/build_ai_autoresearch --extra training
```

The reference model is
[SmolVLM2-256M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct/tree/067788b187b95ebe7b2e040b3e4299e342e5b8fd),
pinned to an immutable revision. Although video-capable, it receives one image
per example. Evaluation reuses HFlow's public wearer-hand prompt, EXIF-oriented
RGB, 512-pixel processing with image splitting disabled, greedy decoding, and
at most four output tokens. Only a stripped `0`, `1`, or `2` is a valid answer;
invalid answers stay in the metric denominator. This is an explicitly named
Transformers CPU reference backend, with a required recorded reason.

Freeze a protocol. The reason is a positional argument:

```bash
uv run --locked --project examples/build_ai_autoresearch --extra training \
  python -m examples.build_ai_autoresearch.experiment init \
  data/build-ai-autoresearch/prepared data/build-ai-autoresearch/experiment \
  'CPU reference pilot before allocating GPU compute' \
  --train-samples 192 --development-samples 48 --confirmation-samples 48 \
  --training-seconds 300 --max-trials 2
```

Each trial gets the same **300-second wall-clock allowance on the same host**,
with the same four CPU threads. Fixed base-model loading and its initial audit
are outside this allowance and have a separate 120-second startup timeout.
Training-file import, adapter/optimizer setup, image processing, updates, final
base-weight auditing, and adapter saving are inside it. Development evaluation
runs afterwards under the fixed evaluator. Faster algorithms can perform more
updates within the allowance; a reported-step safety cap defaults to 1,024.
The trial records actual time and completed-step losses. The default loop leaves
five seconds for auditing/saving; allow more if a proposed step is expensive.
The parent watchdog kills an over-budget worker and marks the trial failed.

Defaults permit at most eight trials, 192 training frames, and 48 frames each
for development and confirmation. Selection is deterministic for seed 42.
Training reserves one frame per class, then follows the original sample pool.
Development/confirmation choose classes in round-robin order; their scores
reflect those subsets, not the original class prior. A small preparation pilot
may lack a class; initialization rejects that. For an execution smoke check,
use smaller frame counts and `--training-seconds 15`; this is not a quality run.

Run the untouched baseline, then the naive training code copied into the
experiment directory:

```bash
uv run --locked --project examples/build_ai_autoresearch --extra training \
  python -m examples.build_ai_autoresearch.experiment baseline-transformers-reference \
  data/build-ai-autoresearch/experiment
uv run --locked --project examples/build_ai_autoresearch --extra training \
  python -m examples.build_ai_autoresearch.experiment trial-transformers-reference \
  data/build-ai-autoresearch/experiment
```

## One editable training file

The coding agent edits **only `<experiment>/train.py`**, which defines:

```python
from peft import PeftModel
from examples.build_ai_autoresearch.training_api import TrainingContext


def train(
    context: TrainingContext,
) -> PeftModel: ...  # Train LoRA on context.base_model; return the adapted model.
```

[The initial training file](./train.py) is a naive LoRA loop: language-model
query/value projections, rank 4, alpha 8, no dropout, AdamW at 0.0002, uniform
sampling, batch size one, and gradient clipping at 1. It is runnable without an
agent. A person can also edit it and run experiments manually.

Unlike a JSON parameter sweep, the agent can implement new training behavior:

- Change adapter placement and which adapters receive updates.
- Design class-aware, curriculum, or difficulty-based training sampling.
- Add full-frame, count-preserving training augmentations such as brightness.
- Implement loss weighting, optimizer changes, or learning-rate schedules.
- Change batching and training-loop efficiency within the same time allowance.

`context.samples`, `context.seed`, `context.max_steps`, and
`context.remaining_seconds` describe the fixed training inputs and limits.
`context.batch(sample, transform=...)` verifies image identity, renders the
fixed prompt/answer, and masks the exact prompt prefix and padding. Report one
finite loss per optimizer update with `context.record_loss(...)`. Loss weighting
may change the objective; teacher targets and their identities must stay fixed.
Augmentations must preserve the count label: avoid crops or edits that remove
hands. Evaluation always uses the original images and frozen preprocessing.

The base architecture/weights, public model revision, task, prompt, splits,
scoring, dependencies, and compute allowance are fixed. The worker verifies
that base parameters remain unchanged and accepts only LoRA adapters with no
full-module or bias updates. Changes to base architecture/weights are rejected.
Code changes can therefore explore training algorithms while the exported
artifact remains an adapter on the pinned base model.

## Agent search and frozen confirmation

Give your coding agent [program.md](./program.md), the experiment path, and the
trial command above. It reads development failures, states a hypothesis, edits
training code, runs a trial, and keeps useful changes. The agent is useful for
implementing strategies that were not enumerated in a configuration schema;
no paid model API or particular agent is built into the runner.

Each attempt snapshots `train.py` into its trial directory before execution.
The worker runs that snapshot in a separate process. The evaluator then loads
only its saved adapter on fresh pinned base weights: candidate code is never
imported into the evaluator process. Source hashes, checkpoint hashes, runtime
identities, losses, and fixed-subset predictions are bound into the report.
Previous source snapshots remain selectable after the working file changes.

Trials report fixed three-class macro-F1, accuracy, invalid outputs, per-class
support/F1, and per-corpus agreement. A cross-corpus duplicate appears in each
source-corpus breakdown, but only once overall. Selection uses strict macro-F1
improvement; ties keep the earlier result, including the untouched baseline.
Failed/interrupted/over-budget attempts count. Run commands serially, on the
same host without competing training jobs, and do not change hardware or threads
mid-search. Wall-clock comparisons remain subject to ordinary host-load noise.

The operator ends search and confirms once:

```bash
uv run --locked --project examples/build_ai_autoresearch --extra training \
  python -m examples.build_ai_autoresearch.experiment freeze \
  data/build-ai-autoresearch/experiment
uv run --locked --project examples/build_ai_autoresearch --extra training \
  python -m examples.build_ai_autoresearch.experiment confirm-transformers-reference \
  data/build-ai-autoresearch/experiment data/build-ai-autoresearch/prepared
```

Initialization copies only training/development images. It records the test
manifest digest without opening that partition. **The worker is not a security
sandbox.** Editable Python executes with its process's permissions. For agent
runs, use an isolated account/container with fixed code, dependencies, protocols,
receipts, and source snapshots read-only to the editing agent. Allow the agent
to write only the working `train.py` and research notes, and have the operator
invoke the runner. Keep prepared/source/test data and the confirmation command
outside the agent/worker filesystem. Hashes detect accidental drift; permissions
and isolation enforce access. Candidate code must not patch runtime modules,
forge receipts, alter timers, or inspect development/test images during training.

Selection is written exclusively and blocks further trials. Confirmation
creates an exclusive directory before opening the test partition; a failed attempt also
consumes confirmation. It materializes test images there, so keep that directory
away from the search agent. Do not resume search on the same test set after
viewing confirmation. A failed or disappointing result is a result.

Changed evaluator code, lockfiles, images, selected source snapshots, and
checkpoints are rejected. Interrupted directories without final receipts are
incomplete. Protocol schema 2 uses time-bounded editable code; old JSON-recipe/
fixed-step experiments must be kept as historical evidence and cannot resume
under this runner. Initialize a new experiment instead of rewriting receipts.

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
a separately provisioned OpenAI-compatible VLM endpoint, then follow the existing
[HFlow Build AI evaluation guide](../../docs/how-to/run-build-ai-evaluation.md).
Endpoint/model support must be checked for your serving stack; this CPU example
does not validate vLLM deployment. The published evaluation pool overlaps this
training pool, so a rerun on it is an integration check, not independent quality
evidence. The guide also runs a **different input/output contract**: its route
puts text before the image and requests JSON by default, while this experiment
puts the image first and scores bare digits. Its serving-side image processing
can also differ. Exporting weights does not make that route equivalent to this
experiment. Use this example's confirmation command for its reported metric;
treat the linked route as a separate integration test with separately recorded
prompt, message order, processor, and response-format settings.
Teacher agreement and exact-frame separation do not establish
production readiness or recording-independent generalization.

## CPU pilot evidence

The earlier fixed-step prototype established development headroom: 192 training
frames, 48 balanced development frames, 256 steps, learning rate 0.0002, rank 8,
and class-balanced sampling improved macro-F1 from 0.1667 to 0.4889. Training
took approximately 268 seconds on a four-thread CPU. One-hand class F1 remained
zero. This is historical development evidence for the public task/model, not
confirmation or a comparison of the new time-budgeted training strategies.

The editable-code runner is checked separately with short public-data CPU
trials and two distinct source snapshots under the same 15-second allowance
(10 and 8 updates), plus baseline/selection/one-time-confirmation/export. Both
short trials stayed at macro-F1 0.1667. A typed training module also loaded
successfully; a deliberate base-weight mutation was rejected and consumed an
attempt. Real-process tests cover completion and over-budget termination.
These establish execution and budget/source contracts; short smoke checks do not establish quality. All frames,
predictions, source snapshots, and weights remain under ignored `data/`.

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
