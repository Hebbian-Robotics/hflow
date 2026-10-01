# Build AI tiny-VLM autoresearch: data preparation

This example is being developed in stages. **The implemented stage prepares
the public training data; training, model evaluation, and the agent loop are
not implemented yet.** It builds on the existing
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

## Planned research cycle

The next stage will use a fixed public tiny VLM and fixed prompts, image
processing, decoding, seed, and per-trial training budget:

1. Measure the untouched model and a naive LoRA fine-tune on development data.
2. Permit the agent to edit one small recipe containing learning rate, LoRA
   rank, and class-balancing choice. Freeze everything else.
3. Train from the same base weights each time, evaluate teacher agreement with
   macro-F1 plus per-class/per-corpus counts, and keep only improvements.
4. Freeze the selected recipe and checkpoint identities, then run test once.
   Keep confirmation data inaccessible to the editing agent.
5. Use the trained checkpoint through the existing HFlow Build AI check.

An agent loop is useful only after the first baseline and fine-tuning pilot
establish measurable headroom and affordable runtime. The frame-level
confirmation limitations above remain visible in every eventual report.

Validation:

```bash
uv run --locked --project examples/build_ai_autoresearch \
  pytest -q examples/build_ai_autoresearch/tests
```
