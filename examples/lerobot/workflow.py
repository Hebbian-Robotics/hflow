"""A bounded, reproducible real-camera LeRobot workflow (hflow issue #191).

Composes the public LeRobot adapter commands and callables end to end for
one pinned corpus: import a bounded episode subset with every declared
camera, run ``hflow doctor`` on the converted inputs, process the episodes
through an :class:`hflow.App` with the applicable default checks and the
built-in contact-sheet enrichment, cut a training selection with the
documented curation policy, export the selection as both an HFlow dataset
snapshot and a loadable LeRobot Dataset v3 repository, and verify the
export in a clean process.

The pinned corpus and the bounded episode subset live in
``source-manifest.json`` next to this file; all downloaded media and
generated artifacts stay under the gitignored data directory, so re-running
without source or configuration changes is safe and reproduces the same
selection.

    uv run python examples/lerobot/workflow.py [--data-dir DATA_DIR]

Prerequisites: network access to the Hugging Face Hub, enough disk for the
pinned corpus and generated artifacts (~1 GB), and the repository's normal
development environment (managed FFmpeg included). The import step converts
only the listed episodes; the export step additionally materializes the
full pinned source archive through the public importer, so the first run
takes roughly 30-60 minutes of video conversion on a laptop.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

import hflow
from hflow.cli import main as cli_main
from hflow.curation import curate

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.lerobot.export import export  # noqa: E402


def curation_policy_sql(cameras: Sequence[str]) -> str:
    """The documented curation policy as SQL over the catalog's episodes view.

    Keeps every episode whose recorded quality evidence shows a canonical
    status of ``ok``, every declared camera fully decoded
    (``decoded_frame_count`` equals ``message_count``), no frame freezes
    (``freeze_total_s`` of zero), and no timestamp-regularity violations
    (``period_violation_pct`` of zero). Coverage denominators ride on the
    manifest as scalar columns, so a reader sees the policy's reach, not
    just the survivors. Selection is ordered by episode id, making the
    manifest deterministic across runs.

    Measurement keys are per-channel: the canonical channel topic for a
    LeRobot camera is ``/<camera key>`` (the converter's own contract), and
    each quality metric is recorded under that topic.
    """

    def quoted(camera: str, metric: str) -> str:
        return f'"/{camera}/{metric}"'

    measured = " AND ".join(
        f"{quoted(camera, 'decoded_frame_count')} IS NOT NULL" for camera in cameras
    )
    fully_decoded = " AND ".join(
        f"{quoted(camera, 'decoded_frame_count')} = {quoted(camera, 'message_count')}"
        for camera in cameras
    )
    no_freeze = " AND ".join(
        f"coalesce({quoted(camera, 'freeze_total_s')}, 0) = 0" for camera in cameras
    )
    no_violation = " AND ".join(
        f"coalesce({quoted(camera, 'period_violation_pct')}, 0) = 0" for camera in cameras
    )
    return f"""\
SELECT
  episode_id,
  metadata_json,
  (SELECT COUNT(*) FROM episodes) AS total_episodes,
  (SELECT COUNT(*) FROM episodes WHERE status = 'ok') AS ok_episodes,
  (SELECT COUNT(*) FROM episodes WHERE {measured}) AS fully_measured_episodes
FROM episodes
WHERE status = 'ok' AND {fully_decoded} AND {no_freeze} AND {no_violation}
ORDER BY episode_id"""


def run_import(data_dir: Path, manifest: dict) -> tuple[list[Path], dict]:
    """Convert the pinned subset and verify the resolved commit.

    The importer's contract is one episode per call; the resume logic skips
    episodes whose published identity already matches this import
    (resolved commit, source episode, cameras, converter version, GOP), so
    looping over the subset is also what makes later runs cheap.
    """
    uris: list[Path] = []
    for index in (int(index) for index in manifest["episodes"]):
        uris.extend(
            Path(uri)
            for uri in hflow.import_lerobot_dataset(
                dataset_repo=manifest["repository"],
                revision=manifest["revision"],
                output_dir=data_dir,
                episode_index=index,
                camera_keys=tuple(manifest["cameras"]),
            )
        )
    prepared = json.loads((data_dir / "prepared-manifest.json").read_text())
    resolved = prepared["dataset"]["revision"]
    if resolved != manifest["revision"]:
        raise RuntimeError(
            f"import resolved to {resolved}, expected the pinned revision "
            f"{manifest['revision']}; refusing to continue"
        )
    deduplicated = list(dict.fromkeys(Path(uri) for uri in uris))
    print(
        f"import: converted {len(deduplicated)} episode(s) from "
        f"{manifest['repository']} @ {resolved[:12]}"
    )
    return deduplicated, prepared


def run_doctor(landing_uris: Sequence[Path | str]) -> None:
    """Every converted input must conform to the canonical-episode convention."""
    exit_code = cli_main(["doctor", *(str(uri) for uri in landing_uris)])
    if exit_code != 0:
        raise RuntimeError(f"hflow doctor refused the converted inputs (exit {exit_code})")
    print(f"doctor: {len(landing_uris)} converted input(s) conforming")


def run_process(data_dir: Path, landing_uris: Sequence[Path | str]) -> None:
    """Process every episode with the default checks and contact sheets."""
    app = hflow.App("lerobot-workflow", data_root=str(data_dir))
    for uri in landing_uris:
        app.process(uri)


def run_curation(data_dir: Path, cameras: Sequence[str]) -> Path:
    """Cut the training selection from the recorded quality evidence."""
    sql = curation_policy_sql(cameras)
    manifest_path = data_dir / "manifest.parquet"
    report = curate(data_dir / "catalog", sql, output=manifest_path)
    print(report.summary())
    print(f"curation: wrote {report.row_count} selected episode(s) to {manifest_path.name}")
    return manifest_path


def run_snapshot(data_dir: Path, manifest_path: Path) -> Path:
    """Write the tool-neutral snapshot with copied media."""
    snapshot_dir = data_dir / "snapshot"
    exit_code = cli_main(
        [
            "export",
            "snapshot",
            "--catalog",
            str(data_dir / "catalog"),
            "--manifest",
            str(manifest_path),
            "--output",
            str(snapshot_dir),
            "--media",
            "copy",
            "--overwrite",
        ]
    )
    if exit_code != 0:
        raise RuntimeError(f"hflow export snapshot failed (exit {exit_code})")
    print(f"snapshot: wrote {snapshot_dir}")
    return snapshot_dir


def run_v3_export(data_dir: Path, manifest_path: Path, cameras: Sequence[str]) -> Path:
    """Export the selection as a local LeRobot Dataset v3 repository."""
    destination = data_dir / "v3"
    export(
        destination=destination,
        manifest=manifest_path,
        sql=None,
        camera_keys=tuple(cameras),
    )
    print(f"v3 export: wrote {destination}")
    return destination


def run_verify(v3_dir: Path, expected_episodes: Sequence[int]) -> None:
    """Verify the exported repository in a clean process (no HFlow import)."""
    verify_script = Path(__file__).with_name("verify.py")
    subprocess.run(
        [
            sys.executable,
            str(verify_script),
            str(v3_dir),
            "--expect",
            ",".join(str(index) for index in expected_episodes),
        ],
        check=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the pinned real-camera LeRobot workflow end to end."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data/lerobot_workflow"),
        help="working directory for all downloads and generated artifacts",
    )
    args = parser.parse_args(argv)

    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads(Path(__file__).with_name("source-manifest.json").read_text())
    cameras = tuple(source_manifest["cameras"])
    subset = [int(index) for index in source_manifest["episodes"]]

    print(f"workflow: {source_manifest['repository']} @ {source_manifest['revision']}")
    print(f"  subset: {subset}")
    print(f"  cameras: {', '.join(cameras)}")

    try:
        landing_uris, prepared = run_import(data_dir, source_manifest)
        run_doctor(landing_uris)
        run_process(data_dir, landing_uris)

        manifest_path = run_curation(data_dir, cameras)
        snapshot_dir = run_snapshot(data_dir, manifest_path)
        v3_dir = run_v3_export(data_dir, manifest_path, cameras)

        selected = pq.read_table(manifest_path).num_rows
        run_verify(v3_dir, list(range(selected)))

        resolved = prepared["dataset"]["revision"]
        archive_info = json.loads(
            (data_dir / "_lerobot_cache" / resolved / "meta" / "info.json").read_text()
        )
        statuses = _status_counts(data_dir / "catalog")
        print("summary:")
        print(f"  source episodes:        {archive_info.get('total_episodes', 'unknown')}")
        print(f"  immutable revision:     {resolved}")
        print(f"  processed episodes:     {len(landing_uris)}")
        status_line = ", ".join(f"{status}: {count}" for status, count in sorted(statuses.items()))
        print(f"  status counts:          {status_line}")
        print(f"  selected episodes:      {selected}")
        print(f"  curation manifest:      {manifest_path}")
        print(f"  hflow snapshot:         {snapshot_dir}")
        print(f"  lerobot v3 export:      {v3_dir}")
        return 0
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"workflow failed: {error}", file=sys.stderr)
        return 1


def _status_counts(catalog: Path) -> dict[str, int]:
    """Per-status episode counts from the catalog's latest rows."""
    sql = "SELECT status, COUNT(*) AS n FROM episodes GROUP BY status ORDER BY status"
    status_path = catalog.parent / "statuses.parquet"
    curate(catalog, sql, output=status_path)
    conn = duckdb.connect()
    try:
        quoted = str(status_path).replace("'", "''")
        rows = conn.execute(f"SELECT * FROM read_parquet('{quoted}')").fetchall()
    finally:
        conn.close()
    return {str(status): int(count) for status, count in rows}


if __name__ == "__main__":
    raise SystemExit(main())
