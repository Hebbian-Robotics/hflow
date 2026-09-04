# Load a large manifest with memory mapping

**Goal:** start training against a big `manifest.parquet` without every rank
reading the whole file into RAM. Download it once, memory-map it, and assign
each rank its own shard.

At small scale you do not need this: a manifest of thousands of episodes loads
instantly with `pandas.read_parquet`. The pattern matters when the manifest has
tens of millions of rows and many ranks per node, where per-rank in-memory
reads exceed the node's RAM and a network-mounted columnar read pays a request
per scattered seek. HFlow deliberately ships no training dataloader (see
[the architecture](../ARCHITECTURE.md#the-scale-path)); this page is the recipe
for wiring the pattern into your own.

**Prerequisites:** `pyarrow` (the `arrow` extra: `uv sync --extra arrow`), and
a manifest produced by `hflow curate` / `hflow.curate()`.

## The three steps that only work as a set

1. **Download once per node, not per rank.** One rank pulls the manifest from
   object storage onto local node disk with a direct parallel transfer,
   bypassing any network mount. The other ranks wait on a barrier.
2. **Convert once to Arrow IPC, then map.** Parquet pages are compressed, so a
   Parquet read always materializes buffers in RAM. Converting once to an
   **uncompressed Arrow IPC file** makes the bytes on disk the bytes in memory;
   every rank then memory-maps that local copy and only the pages actually
   touched become resident.
3. **Shard on load, zero-copy.** Each rank takes a `slice()` covering its 1/N
   of the rows. On a mapped table a slice is an offset, not a copy -- which is
   why mapping and sharding ship together: slicing an in-memory read has
   already paid for the whole table.

```python
from pathlib import Path

import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq

MANIFEST_URL = "s3://robot-data/manifests/training.parquet"
LOCAL_DIR = Path("/mnt/local-nvme/manifests")


def prepare_on_one_rank() -> Path:
    """Rank 0 (or one rank per node): download, convert to IPC, once."""
    import hflow

    arrow_path = LOCAL_DIR / "training.arrow"
    if arrow_path.exists():  # converted files are immutable; reuse across jobs
        return arrow_path
    parquet_path = hflow.fetch_uri(MANIFEST_URL)  # etag-cached local copy
    table = pq.read_table(parquet_path)
    # Uncompressed on purpose: compressed IPC cannot be mapped zero-copy.
    feather.write_feather(table, arrow_path, compression="uncompressed")
    return arrow_path


def load_my_shard(arrow_path: Path, rank: int, world_size: int) -> pa.Table:
    """Every rank: map the local copy and slice its rows, zero-copy."""
    with pa.memory_map(str(arrow_path)) as source:
        table = pa.ipc.open_file(source).read_all()  # pages stay on disk
    rows_per_rank = (table.num_rows + world_size - 1) // world_size
    return table.slice(offset=rank * rows_per_rank, length=rows_per_rank)
```

Call `prepare_on_one_rank()` from exactly one rank per node, barrier, then
`load_my_shard(...)` from every rank. The `hflow.fetch_uri` download works for
`s3://`, `gs://`, and `az://` URLs with the `bucket` extra installed; any
parallel transfer tool (`aws s3 cp`, `gsutil`, obstore) does the same job.

## Why not just map the Parquet file?

Parquet's footer-first, scattered-seek layout plus per-page compression means a
"mapped" Parquet read still decompresses every touched page into fresh
allocations, and on a network mount each seek is its own request. The IPC
conversion is paid once per manifest per node; every training run on that node
afterwards starts in seconds.

## See also

- [Catalog and curation](../CATALOG.md): producing the manifest this page loads
- [Architecture: the scale path](../ARCHITECTURE.md#the-scale-path): where this
  sits among the deliberately deferred scale mechanisms
