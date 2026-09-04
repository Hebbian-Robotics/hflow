# Import a LeRobot Dataset v3 repository

Use HFlow's built-in importer to turn selected LeRobot Dataset v3 episodes
into canonical MCAP files that `hflow doctor`, `App.process`, Foxglove, and
Rerun can consume. The importer is part of the normal `hflow` installation;
it does not install or import LeRobot, PyTorch, or a Hugging Face SDK.

## Import one episode

Choose a local output directory and name at least one video feature:

```bash
uv run hflow import lerobot \
  --repo lerobot/pusht \
  --revision main \
  --camera observation.image \
  --episode-index 0 \
  --output-dir ./data/lerobot_pusht
```

The command resolves the requested revision to an immutable Hugging Face
commit before downloading data. It writes:

```text
data/lerobot_pusht/
├── _lerobot_cache/                 # downloaded metadata, Parquet, and video
├── landing/
│   └── lerobot_episode_0001.mcap   # canonical episode
└── prepared-manifest.json          # source commit, import summary, and episode receipts
```

Re-running against the same output directory reuses downloaded source files.
Run `uv run hflow doctor ./data/lerobot_pusht/landing/*.mcap` to print the
canonical-format report.

Object-store data roots work the same way when the optional bucket backend is
installed (`uv sync --extra bucket`) and provider credentials are available:

```bash
uv run hflow import lerobot \
  --repo lerobot/pusht \
  --revision main \
  --camera observation.image \
  --episode-index 0 \
  --output-dir gs://robot-data/production
```

Durable outputs land as `landing/*.mcap` and `prepared-manifest.json` under
that prefix. Hugging Face downloads stay in the local mirror under
`_lerobot_cache/` (`HFLOW_MIRROR_DIR`, or `$XDG_CACHE_HOME/hflow/mirrors`) and
are never uploaded into the bucket. The success manifest is published only
after every selected episode object has been written.

`prepared-manifest.json` uses schema version 3. It keeps the dataset source,
selected camera keys, converted episode count, and converter version, and adds
one receipt for every published canonical MCAP:

```json
{
  "schema_version": 3,
  "dataset": {
    "repo_id": "lerobot/pusht",
    "revision": "0123456789abcdef0123456789abcdef01234567",
    "license": "apache-2.0"
  },
  "camera_keys": ["observation.image"],
  "episodes_converted": 1,
  "episodes": [
    {
      "filename": "lerobot_episode_0001.mcap",
      "content_id": "0123456789abcdef",
      "size_bytes": 123456
    }
  ],
  "converter_version": "lerobot-converter-v4"
}
```

Each `content_id` is HFlow's canonical episode content ID (the first 16 hex
characters of the SHA-256 over the published MCAP bytes), and `size_bytes` is
the file size in bytes. `episodes_converted` remains equal to the number of
entries in `episodes`, so readers that use the existing import summary fields
can continue to do so while newer readers can identify each delivered file.

For a gated or private repository, export a read token as `HF_TOKEN` (or
`HUGGING_FACE_HUB_TOKEN`) before running the command. HFlow sends the token
only to Hugging Face requests and never records it in an episode or manifest.

## Import multiple cameras

Repeat `--camera` for each Dataset v3 video feature:

```bash
uv run hflow import lerobot \
  --repo lerobot/svla_so101_pickplace \
  --revision f641879e22172be7e8161d5e6c1503c2d2feb657 \
  --camera observation.images.up \
  --camera observation.images.side \
  --episode-index 0 \
  --output-dir ./data/lerobot_svla
```

Omit `--episode-index` to import every episode. Each selected camera becomes
one `foxglove.CompressedVideo` channel; state and action become ROS 2 CDR
channels named `/observation.state` and `/action`.

## Supported Dataset v3 subset

| Feature | Support |
|---|---|
| Video features with `dtype: video` | Supported; select them with `--camera` |
| One-dimensional fixed-width `float32` observation state | Supported |
| One-dimensional fixed-width `float32` action | Supported |
| Image-only features | Not supported |
| Language tensors, depth maps, or arbitrary nested features | Not supported |

The importer reads `meta/info.json`, the episode index, Parquet feature
columns, frame rate, episode boundaries, and video-path templates. It refuses
an unsupported or missing required feature before publishing an episode.

Every output records the source repository, resolved commit, source episode
index, task, embodiment, importer version, and FFmpeg build. Video is encoded
without B-frames so every source frame survives HFlow's Annex B-to-MP4 access
path.

## Call the Python API

The installed package exports the same operation:

```python
from pathlib import Path

import hflow

episodes = hflow.import_lerobot_dataset(
    dataset_repo="lerobot/pusht",
    revision="main",
    camera_keys=("observation.image",),
    episode_index=0,
    output_dir=Path("./data/lerobot_pusht"),
)
```

The return value is the list of published episode URIs under
`landing/` -- absolute path strings for a local data root, or `s3://` /
`gs://` / `az://` object URIs for a bucket root.

This command imports source data into HFlow. It does not upload data, train a
policy, or export a curated HFlow manifest back to LeRobot Dataset v3.
