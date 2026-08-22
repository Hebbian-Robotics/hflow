"""Workspace builders shared by the hflow-server suite.

A uniquely-named module (never ``conftest``) so test modules can import the
types under any pytest import mode without basename collisions against the
repository's root test conftest.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import pytest

import hflow
from hflow.catalog import Catalog, CheckRunRow
from hflow.transform import EpisodeStamps

PIPELINE_VERSION = "pipeline0000001"

STAMPS = EpisodeStamps(
    schema_version="1",
    pipeline_version=PIPELINE_VERSION,
    ffmpeg_version="ffmpeg version test",
    robot_software_version="sim-0.1.0",
)


@dataclass(frozen=True)
class PopulatedWorkspace:
    """One data root with four episodes covering the whole API surface."""

    data_root: Path
    ok_episode_id: str  # fold_napkin/alice: two runs, media, measurements
    quarantined_episode_id: str  # pour_water/bob: quarantined with tags
    escaping_episode_id: str  # fold_napkin/carol: canonical + media OUTSIDE the root
    minimal_episode_id: str  # stack_blocks: no operator, no checks
    contact_sheet_file: Path
    outside_media_file: Path
    canonical_file: Path


def build_populated_workspace(tmp_path_factory: pytest.TempPathFactory) -> PopulatedWorkspace:
    data_root = tmp_path_factory.mktemp("ui-data-root")
    outside_directory = tmp_path_factory.mktemp("outside-the-root")
    episodes_directory = data_root / "episodes"
    episodes_directory.mkdir()
    media_directory = data_root / "media"
    media_directory.mkdir()
    catalog = Catalog(data_root / "catalog")

    contact_sheet_file = media_directory / "wrist_cam.jpg"
    contact_sheet_file.write_bytes(b"\xff\xd8\xff\xe0 fake jpeg body \xff\xd9")
    outside_media_file = outside_directory / "leaked.jpg"
    outside_media_file.write_bytes(b"\xff\xd8\xff\xe0 outside the root \xff\xd9")

    def joint_check_row(max_velocity: float) -> CheckRunRow:
        return CheckRunRow(
            check_name="joint_check",
            check_version="v1",
            critical=False,
            status=hflow.CheckStatus.MEASURED,
            duration_s=0.01,
            measurements={
                "max_velocity": max_velocity,
                # JSON-illegal doubles: the API must null these, not crash.
                "nan_metric": float("nan"),
                "inf_metric": float("inf"),
            },
            tags=["seen"],
            intervals=[hflow.Interval(start_ns=0, end_ns=100, label="span")],
        )

    contact_sheet_row = CheckRunRow(
        check_name="media/contact_sheet",
        check_version="v1",
        critical=False,
        status=hflow.CheckStatus.MEASURED,
        duration_s=0.01,
        measurements={"artifact/wrist_cam": str(contact_sheet_file)},
    )

    canonical_file = episodes_directory / "fold_a.canonical.mcap"
    canonical_file.write_bytes(b"canonical fold napkin A")
    ok_metadata = {
        "task": "fold_napkin",
        "operator": "alice",
        "success": "true",
        "embodiment": "arm-1",
    }
    first_append = catalog.append_episode(
        canonical_path=canonical_file,
        stamps=STAMPS,
        episode_metadata=ok_metadata,
        check_rows=[joint_check_row(1.5), contact_sheet_row],
    )
    # Distinct recorded_at so "latest run" ordering stays deterministic.
    time.sleep(0.01)
    second_append = catalog.append_episode(
        canonical_path=canonical_file,
        stamps=STAMPS,
        episode_metadata=ok_metadata,
        check_rows=[joint_check_row(2.0), contact_sheet_row],
    )
    assert first_append.written and second_append.written
    assert first_append.episode_id == second_append.episode_id
    time.sleep(0.01)

    quarantined_canonical = episodes_directory / "pour_b.canonical.mcap"
    quarantined_canonical.write_bytes(b"canonical pour water B")
    quarantined_append = catalog.append_episode(
        canonical_path=quarantined_canonical,
        stamps=STAMPS,
        episode_metadata={
            "task": "pour_water",
            "operator": "bob",
            "success": "false",
            "embodiment": "arm-1",
        },
        check_rows=[
            CheckRunRow(
                check_name="camera_blackout",
                check_version="v1",
                critical=True,
                status=hflow.CheckStatus.FAILED,
                duration_s=0.02,
                measurements={"black_pct": 80.0},
            )
        ],
        quarantine_tags=["failed:camera_blackout"],
    )
    time.sleep(0.01)

    escaping_canonical = outside_directory / "escape_c.canonical.mcap"
    escaping_canonical.write_bytes(b"canonical outside the data root C")
    escaping_append = catalog.append_episode(
        canonical_path=escaping_canonical,
        stamps=STAMPS,
        episode_metadata={"task": "fold_napkin", "operator": "carol", "embodiment": "arm-2"},
        check_rows=[
            CheckRunRow(
                check_name="media/contact_sheet",
                check_version="v1",
                critical=False,
                status=hflow.CheckStatus.MEASURED,
                duration_s=0.01,
                measurements={
                    "artifact/outside": str(outside_media_file),
                    "artifact/missing": str(media_directory / "never_written.jpg"),
                },
            )
        ],
    )
    time.sleep(0.01)

    minimal_canonical = episodes_directory / "stack_d.canonical.mcap"
    minimal_canonical.write_bytes(b"canonical stack blocks D")
    minimal_append = catalog.append_episode(
        canonical_path=minimal_canonical,
        stamps=STAMPS,
        episode_metadata={"task": "stack_blocks"},
        check_rows=[],
    )

    return PopulatedWorkspace(
        data_root=data_root,
        ok_episode_id=first_append.episode_id,
        quarantined_episode_id=quarantined_append.episode_id,
        escaping_episode_id=escaping_append.episode_id,
        minimal_episode_id=minimal_append.episode_id,
        contact_sheet_file=contact_sheet_file,
        outside_media_file=outside_media_file,
        canonical_file=canonical_file,
    )
