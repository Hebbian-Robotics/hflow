# Export a portable review dataset

Use this export when a person or a downstream tool needs the catalog snapshot,
quality evidence, and preview artifacts together. The result is a local
directory of standard Parquet tables and ordinary media files. It has no
required viewer and no HFlow-specific Python object to deserialize.

## Export the selected episodes

First create the selection with the normal curation interface:

```bash
uv run hflow curate \
  "SELECT episode_id FROM episodes WHERE status = 'ok'" \
  --output ./data/review-manifest.parquet
```

Then export it. Reference mode is fast and preserves the artifact locations
already recorded in the catalog:

```bash
uv run hflow export review \
  --manifest ./data/review-manifest.parquet \
  --output ./data/review-export \
  --media references
```

Use copy mode for a self-contained directory that can move to another machine:

```bash
uv run hflow export review \
  --manifest ./data/review-manifest.parquet \
  --output ./data/review-export \
  --media copy
```

Omit `--manifest` to export every latest episode in the catalog. An existing
output directory is refused by default; pass `--overwrite` to replace it only
after the new export has been staged completely.

The observable result is:

```text
review-export/
├── format.json
├── episodes.parquet
├── measurements.parquet
├── media.parquet
├── check_runs.parquet
├── tags.parquet
├── intervals.parquet
└── assets/                 # copy mode only, when artifacts exist
```

## Format contract

`format.json` identifies `hflow-review-dataset` format version `1`, names each
table, records the media mode, and states whether media paths are relative to
the export directory. The JSON marker is written last and the completed
directory is activated atomically.

The Parquet tables form one snapshot:

| File | Grain and purpose |
| --- | --- |
| `episodes.parquet` | One row per selected episode. It contains identity, canonical/source URI, promoted metadata, version stamps, status, and wide numeric or boolean measurements. |
| `measurements.parquet` | One row per latest `(episode_id, key)`, with separate number, text, and boolean value columns. This preserves text labels and typed evidence that does not belong in the wide table. |
| `media.parquet` | One row per artifact measurement: artifact name, producing step and version, role, inferred media kind and MIME type, URI, and timestamp. |
| `check_runs.parquet` | The latest run of each producing step per episode, including status, duration, and error. |
| `tags.parquet` | Tags belonging to those latest step runs. |
| `intervals.parquet` | Time intervals belonging to those latest step runs. |

In `references` mode, `media.parquet.uri` is the original local path or object
store URL. In `copy` mode, it is a POSIX path below `assets/`, relative to the
export directory. Copy mode downloads bucket-backed artifacts through HFlow's
normal storage cache and aborts the export if any selected artifact cannot be
materialized.

The library entry point exposes the same behavior:

```python
from pathlib import Path

import hflow

report = hflow.export_review_dataset(
    "./data/catalog",
    Path("./data/review-export"),
    manifest="./data/review-manifest.parquet",
    media_mode=hflow.ReviewMediaMode.COPY,
)
print(report)
```

## Open it without HFlow

DuckDB, pandas, and Polars can all read the output directly:

```python
import duckdb

review_rows = duckdb.sql(
    """
    SELECT episodes.task, episodes.status, media.uri AS preview
    FROM read_parquet('data/review-export/episodes.parquet') episodes
    LEFT JOIN read_parquet('data/review-export/media.parquet') media
      USING (episode_id)
    """
).df()
```

No HFlow server or frontend is involved. HFlow currently ships a JSON workspace
API through `hflow-server`, but it does not ship a browser UI; this export is a
separate interoperability boundary for any review tool.

## Inspect it with Renumics Spotlight

[Renumics Spotlight](https://github.com/Renumics/spotlight) is one optional
consumer. Export with `--media copy`, then install its package and the Parquet
reader in a separate environment:

```bash
uv add pandas pyarrow renumics-spotlight
```

Create `review_with_spotlight.py`:

```python
from pathlib import Path

import pandas as pd
from renumics import spotlight

dataset_directory = Path("data/review-export").resolve()
episodes = pd.read_parquet(dataset_directory / "episodes.parquet")
media = pd.read_parquet(dataset_directory / "media.parquet")

contact_sheets = (
    media.loc[media["role"] == "contact_sheet", ["episode_id", "uri"]]
    .sort_values(["episode_id", "uri"])
    .drop_duplicates("episode_id")
    .rename(columns={"uri": "contact_sheet"})
)
review_rows = episodes.merge(contact_sheets, on="episode_id", how="left")
review_rows["contact_sheet"] = review_rows["contact_sheet"].map(
    lambda relative_path: (
        str(dataset_directory / relative_path) if isinstance(relative_path, str) else relative_path
    )
)

spotlight.show(review_rows, dtype={"contact_sheet": spotlight.Image})
```

Run it from the repository root:

```bash
uv run python review_with_spotlight.py
```

Spotlight opens its own browser viewer over the same pandas DataFrame that any
other consumer can build from the exported Parquet files. The integration is
therefore a documented import recipe, not a HFlow plugin or a Spotlight-only
package.

## See also

- [Query quality evidence and create a manifest](../CATALOG.md)
- [How HFlow fits the robotics data stack](../INTEGRATIONS.md)
- [Serve the workspace JSON API](../SERVE.md)
