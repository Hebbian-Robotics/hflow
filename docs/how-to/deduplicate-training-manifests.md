# Deduplicate a training manifest while retaining provenance

Use `hflow.deduplicate_manifest` to curate a local Parquet manifest using
identities you have already computed. This is a dataset operation, alongside
curation and splitting. HFlow's `@app.check` and `@app.enrich` decorators process
individual episodes; a check can record a content digest, while curation
compares that evidence across samples.

No MCAP conversion, model, API key, GPU, or new optional dependency is required.

## Declare equivalence and conflicts

Give every source occurrence a unique nonempty string sample ID, even when
multiple occurrences represent the same content. Supply nonempty string identity
columns. Rows are duplicates only when **all** those columns match: a composite
identity is an equality tuple, not the transitive relationship grouping used by
`split_manifest`.

```python
from pathlib import Path
from hflow import ManifestDeduplicationSettings, deduplicate_manifest

report = deduplicate_manifest(
    Path("combined-samples.parquet"),
    Path("deduplicated"),
    settings=ManifestDeduplicationSettings(
        sample_id_column="sample_id",
        identity_columns=("pixel_sha256",),
        conflict_columns=("hand_count",),
    ),
)
```

The caller owns media decoding and the meaning of each identity. For example,
a digest of EXIF-oriented RGB pixels detects re-encoded copies with identical
pixels; a digest of encoded bytes has different semantics. Neither establishes
near-duplicate, recording, or participant independence.

Within each identity group, HFlow chooses the lexicographically smallest sample
ID. Selection is independent of input row order. Other columns come from that
representative, so declare any columns whose disagreement matters as conflict
columns. Differences in unlisted columns are preserved in the member records
but do not constitute a reported conflict. Null is a distinct value: null
versus a known label is a conflict; all-null labels agree but remain missing.
HFlow does not infer which label is correct or apply a majority vote.

Ambiguous source column names (ignoring case), missing columns, invalid identity types, null/empty identities, repeated sample
IDs, and reserved output column names are rejected. An empty input is rejected.

## Inspect evidence and apply your policy

The output contains:

- `samples.parquet`: one representative per identity group, including groups
  with conflicting labels. Original columns are retained, with an added
  `deduplication_conflicts` list naming configured columns that disagree.
- `members.parquet`: **every original row and column**, with an added
  `retained_sample_id` referring to its representative. Source provenance and
  recording associations survive removal of copies from the training pool.
- `receipt.json`: settings, algorithm/schema and runtime versions, input
  snapshot digest, output digests, and input/unique/conflicting sample counts.

`deduplication_conflicts` and `retained_sample_id` are reserved input names.
The report's `unique_samples` count includes conflicting groups.

A policy that excludes contradictory labels can use DuckDB:

```python
import duckdb

with duckdb.connect() as connection:
    connection.read_parquet("deduplicated/samples.parquet").filter(
        "len(deduplication_conflicts) = 0"
    ).write_parquet("training-samples.parquet")
```

Exclusion is the caller's explicit choice; it does not erase the original
samples from the deduplication evidence. Keep `members.parquet` and use it to
aggregate source references into your training manifest. When a duplicate links
two recordings, resolve those associations into connected group identities
before splitting. Keeping only the representative's recording ID loses that
relationship and can allow leakage.

Then [split the training manifest](./split-training-manifests.md). The
[Build AI example](../../examples/build_ai_autoresearch/README.md) demonstrates
pixel fingerprinting, provenance aggregation, explicit hand-label exclusions,
and frozen frame-level splits.

## Freeze the result

HFlow reads a private input snapshot so its digest identifies processed bytes.
DuckDB groups and writes the manifest columns without loading media. Resource
use still depends on the number and size of rows; no bounded-memory guarantee
is provided. The API operates on local files; downloading and publishing bucket
objects remain caller-owned.

The output directory must not exist. Ordinary failures remove the newly created
output. The receipt is written last; process termination can leave incomplete
files. Wait for successful return, parse the receipt, and verify its hashes
before consuming an output. Publication of the directory is not atomic.
Parquet bytes may differ across DuckDB versions. Freeze completed artifacts;
do not rewrite historical receipts when code or data changes.
