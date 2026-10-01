"""Split synthetic labelled samples by recording and duplicate relationships.

Run: uv run python examples/split_training_manifest.py
Writes synthetic Parquet and split receipts under data/manifest-splits-demo.
No media, model, network, API key, or GPU is needed.
"""

from pathlib import Path

import duckdb

from hflow import ManifestSplitSettings, split_manifest


def main() -> None:
    output_root = Path("data/manifest-splits-demo")
    output_root.mkdir(parents=True, exist_ok=False)
    source_manifest = output_root / "samples.parquet"
    # These rows represent samples after media deduplication. Recording B's
    # final sample shares a discovered duplicate group with recording A.
    with duckdb.connect() as connection:
        connection.execute(
            "CREATE TABLE samples (sample_id VARCHAR, recording_id VARCHAR, duplicate_group VARCHAR, teacher_label INTEGER)"
        )
        connection.executemany(
            "INSERT INTO samples VALUES (?, ?, ?, ?)",
            [
                (
                    f"sample-{index:03}",
                    f"recording-{index // 2:02}",
                    "shared" if index in (1, 2) else f"image-{index:03}",
                    index % 3,
                )
                for index in range(42)
            ],
        )
        connection.sql("SELECT * FROM samples").write_parquet(str(source_manifest))
    report = split_manifest(
        source_manifest,
        output_root / "splits",
        settings=ManifestSplitSettings(
            sample_id_column="sample_id",
            group_columns=("recording_id", "duplicate_group"),
            seed=42,
        ),
    )
    for partition in report.partitions:
        print(f"{partition.name}: {partition.row_count} samples in {partition.group_count} groups")
    print(f"Receipt: {report.output_directory / 'receipt.json'}")


if __name__ == "__main__":
    main()
