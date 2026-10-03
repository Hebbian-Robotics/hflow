# Split a training manifest without separating related samples

Use `hflow.split_manifest` to produce reproducible partitions of a local
Parquet manifest. It accepts episode manifests or downstream sample manifests;
no MCAP conversion, catalog, model, API key, or GPU is required.

## Prepare your manifest

First combine releases, normalize labels, discover image duplicates, resolve
label conflicts, and retain source provenance. Use
[`hflow.deduplicate_manifest`](./deduplicate-training-manifests.md) to group
declared identities and inspect conflicting labels. Callers own media-specific
identity discovery and conflict policy; the splitting API preserves input rows.

Preserve relationships from removed copies: when a retained image appeared in
two recordings, dropping the second copy must not erase that connection.
Resolve those relationships into connected-group identities before export if
one row cannot represent all source recording identities.

Give every retained sample a unique nonempty string identity. Add nonempty
string columns for each relationship that must not cross partitions, such as
recording identity, participant identity, or a discovered duplicate group.
Namespace identities by their source corpus when unrelated corpora reuse IDs.

Rows sharing a value in **any one** relationship column are connected.
Connections are transitive: if an image relationship joins recordings A and B,
every selected sample from both recordings stays in one partition. Columns
have separate namespaces, so a recording ID equal to an image ID does not
connect them. A known singleton needs its own identity in each group column.

Nulls, empty identities, duplicate sample IDs, missing columns, and non-string
identity columns are rejected before publishing outputs. When original
recording identities are unavailable, grouping only by image identity gives a
frame-level split; it cannot prevent related frames from crossing partitions.
Do not turn unknown recordings into a shared `unknown` group or claim that
unique placeholders prove recording independence.

## Split and freeze

```python
from pathlib import Path
from hflow import ManifestPartition, ManifestSplitSettings, split_manifest

report = split_manifest(
    Path("samples.parquet"),
    Path("frozen-splits"),
    settings=ManifestSplitSettings(
        sample_id_column="sample_id",
        group_columns=("recording_id", "duplicate_group"),
        partitions=(
            ManifestPartition("train", 0.8),
            ManifestPartition("development", 0.1),
            ManifestPartition("test", 0.1),
        ),
        seed=42,
    ),
)
```

At least two named partitions are required. Names match `[a-z][a-z0-9_-]*`;
`assignments` is reserved. Positive finite fractions must sum to one within
an absolute tolerance of 1e-12. Defaults are train/development/test at
0.8/0.1/0.1; two partitions such as search/confirmation also work.

Fractions target **connected-group counts**, not sample counts or class
balance. Groups are ordered by SHA-256 of the seed and their identity digest.
Largest-remainder allocation rounds group quotas; empty quotas borrow from
partitions with more than one group. Every partition receives a whole group,
or the operation rejects a corpus with insufficient independent groups.
Large connected groups can make row proportions very different from requested
fractions. Inspect actual counts and class distributions before training.

Assignments depend on sample identities, relationships, partition order and
fractions, seed, and algorithm version. Input row order does not affect them.
Adding samples or relationships may change assignments. Freeze the input and
splits before searching training recipes; keep the final test partition out
of development decisions. Choose the seed prospectively rather than selecting
it based on model scores.

## Output contract and limits

Each `<name>.parquet` preserves the input schema and selected rows, sorted by
sample identity. `assignments.parquet` maps `sample_id` to `partition` and a
SHA-256 `group_id` derived from sorted member identities.

`receipt.json` records schema/algorithm versions, settings, input snapshot
SHA-256, runtime versions, actual row/group counts, and SHA-256 digests of
every output Parquet file. It contains no input path or labels. Parquet bytes
can vary across DuckDB versions; stable assignment semantics do not imply
identical serialized bytes across runtimes.

The operation reads a private local input copy so the input digest describes
exactly the bytes used. Only identity columns and assignments are held in
Python memory; DuckDB processes the other columns. Memory use grows with the
number of samples and relationships. This is a local-file API; bucket download
and publication remain caller-owned.

The output directory must not exist. Files are written under an exclusively
created directory, with the receipt written last. Ordinary failures remove
that directory. Process termination can leave an incomplete directory: a
directory without a receipt is not a completed split. Readers must wait for
successful return; publication of the directory is not atomic and the receipt
is not an independent verifier of later file changes.

HFlow still ends at curated data and manifests. Model-specific export,
training, and autoresearch orchestration remain downstream; see
[integration boundaries](../INTEGRATIONS.md).
