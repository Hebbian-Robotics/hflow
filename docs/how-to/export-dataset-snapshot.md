# Export a portable dataset snapshot

Use this export when a person or downstream tool needs a selected catalog
snapshot, quality evidence, and media artifacts together. The result is a
local directory of standard Parquet tables and ordinary media files. It has
no required viewer and no HFlow-specific Python object to deserialize.

## Export selected episodes

First create the selection with the normal curation interface:

```bash
uv run hflow curate \
  "SELECT episode_id FROM episodes WHERE status = 'ok'" \
  --output ./data/snapshot-manifest.parquet
```

Then export it. Reference mode is fast and preserves the artifact locations
already recorded in the catalog:

```bash
uv run hflow export snapshot \
  --manifest ./data/snapshot-manifest.parquet \
  --output ./data/dataset-snapshot \
  --media references
```

Use copy mode for a self-contained directory that can move to another machine:

```bash
uv run hflow export snapshot \
  --manifest ./data/snapshot-manifest.parquet \
  --output ./data/dataset-snapshot \
  --media copy
```

Omit `--manifest` to export every latest episode in the catalog. An existing
output directory is refused by default. `--overwrite` replaces it only when
its `format.json` identifies a supported HFlow dataset snapshot, and only
after the new export has been staged completely. Unmarked directories and
snapshots with unknown format versions are never removed.

The observable result is:

```text
dataset-snapshot/
├── format.json
├── samples.parquet
├── measurements.parquet
├── observations.parquet
├── media.parquet
├── check_runs.parquet
├── tags.parquet
├── intervals.parquet
└── assets/                 # copy mode only, when artifacts exist
```

## Format contract

`format.json` identifies `hflow-dataset-snapshot` format version `1`, names
each table (a name-to-filename map), records the media mode, and states whether
media paths are relative to the export directory. The JSON marker is written
last and the completed directory is activated atomically.

Under the same format version `1`, an additive `integrity` block records
delivery integrity so a later verifier (not shipped here) can tell whether the
published bytes are still intact. The original `tables` map is unchanged for
external readers; receipts live only under `integrity`:

| Field | Meaning |
| --- | --- |
| `tables.<name>` | Required Parquet filename (unchanged string map) |
| `integrity.tables.<name>.path` | Same file relative to the export root |
| `integrity.tables.<name>.size_bytes` | Size of that file at export time |
| `integrity.tables.<name>.sha256` | Full SHA-256 of that file's bytes |
| `integrity.assets[]` | Same `path` / `size_bytes` / `sha256` for every regular file under `assets/` in copy mode; empty in references mode (remote media are not fetched) |
| `integrity.content_id` | Full SHA-256 of the normalized inventory (all table and asset receipts, sorted by path), so a deleted member is visible even when every remaining file still matches |

Copy mode re-reads each copied asset once after the copy to compute its hash
(a second full read of media bytes on export).

This is a receipt, not a verify command: export does not re-read the
destination after transfer, and there is no public `verify_dataset_snapshot`
API or CLI yet. Older HFlow overwrite checks only `format` and
`format_version`, and readers that only consume the string `tables` map keep
working; the `integrity` key is purely additive under format version `1`.

The receipt travels unsigned inside the `format.json` it describes, so it
catches corruption and accidental loss, not tampering: anyone who can edit a
table can recompute the hashes to match. Use a signature or an out-of-band
checksum if you need to detect a deliberate change.

The Parquet tables form one snapshot:

| File | Grain and purpose |
| --- | --- |
| `samples.parquet` | The generic entry point: one row per selected episode, with identity, canonical/source URI, promoted metadata, version stamps, status, wide numeric or boolean measurements, and nullable `media_*` columns for one representative artifact. |
| `measurements.parquet` | One row per latest `(episode_id, key)`, with separate number, text, and boolean value columns. This preserves typed evidence that does not belong in the wide table. |
| `observations.parquet` | Timestamped, repeated evidence from each latest check run, one typed row per observation field. |
| `media.parquet` | One row per artifact measurement: artifact name, producing step and version, role, inferred media kind and MIME type, URI, and timestamp. This table preserves every artifact even though `samples.parquet` has one representative artifact per episode. |
| `check_runs.parquet` | The latest run of each producing step per episode, including status, duration, and error. |
| `tags.parquet` | Tags belonging to those latest step runs. |
| `intervals.parquet` | Time intervals belonging to those latest step runs. |

The representative artifact is selected deterministically: contact sheets
first, then images, video, audio, and other artifacts, with artifact name and
URI as tie-breakers. Its fields are `media_uri`, `media_kind`,
`media_mime_type`, `media_role`, and `media_artifact_name`. Consumers that need
multiple cameras or artifact types should join `media.parquet` on `episode_id`.

In `references` mode, media URIs are the original local paths or object-store
URLs. In `copy` mode, they are POSIX paths below `assets/`, relative to the
export directory. Copy mode downloads bucket-backed artifacts through HFlow's
normal storage cache and aborts the export if any selected artifact cannot be
materialized.

The library entry point exposes the same behavior:

```python
from pathlib import Path

import hflow

report = hflow.export_dataset_snapshot(
    "./data/catalog",
    Path("./data/dataset-snapshot"),
    manifest="./data/snapshot-manifest.parquet",
    media_mode=hflow.SnapshotMediaMode.COPY,
)
print(report)
```

If activation succeeds but deleting the previous snapshot fails, the export
still returns successfully because the requested snapshot is already live.
`report.retained_backup` then identifies the backup directory and cleanup
error, and the report summary prints the same warning for CLI users.

## Open it in dataframe tools

DuckDB, pandas, and Polars read `samples.parquet` directly. For a copied
snapshot, resolve non-null `media_uri` values against the snapshot directory
before handing them to a viewer:

```python
from pathlib import Path

import pandas as pd

snapshot_directory = Path("data/dataset-snapshot").resolve()
samples = pd.read_parquet(snapshot_directory / "samples.parquet")
samples["media_uri"] = samples["media_uri"].map(
    lambda media_uri: (
        str(snapshot_directory / media_uri) if isinstance(media_uri, str) else media_uri
    )
)
```

No HFlow server or frontend is involved. HFlow ships a REST workspace API
through `hflow-server`, but it does not ship a browser UI; the snapshot is a
separate interoperability boundary for downstream tools.

## Inspect image samples with Renumics Spotlight

[Renumics Spotlight](https://github.com/Renumics/spotlight) accepts the same
pandas DataFrame. Install its package and the Parquet reader in a separate
environment, run the dataframe snippet above, and add:

```python
from renumics import spotlight

image_samples = samples.loc[samples["media_kind"] == "image"]
spotlight.show(image_samples, dtype={"media_uri": spotlight.Image})
```

This is an import recipe over a generic table, not an HFlow plugin or a
Spotlight-specific export.

## Import image samples into FiftyOne

[FiftyOne](https://github.com/voxel51/fiftyone) models media locations as each
sample's `filepath`. After loading and resolving `samples` as above:

```python
import fiftyone as fo

dataset = fo.Dataset()
for sample_row in samples.loc[samples["media_kind"] == "image"].itertuples():
    sample = fo.Sample(filepath=sample_row.media_uri)
    sample["episode_id"] = sample_row.episode_id
    sample["task"] = sample_row.task
    sample["status"] = sample_row.status
    dataset.add_sample(sample)

session = fo.launch_app(dataset)
session.wait()
```

FiftyOne needs this small field mapping; it does not require another HFlow
package or another export format.

## See also

- [Query quality evidence and create a manifest](../CATALOG.md)
- [How HFlow fits the robotics data stack](../INTEGRATIONS.md)
- [Serve the workspace REST API](../SERVE.md)
