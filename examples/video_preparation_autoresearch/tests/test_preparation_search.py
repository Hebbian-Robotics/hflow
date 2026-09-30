"""Real synthetic media exercises the example's search and confirmation boundaries."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from evaluate import PreparationBudget, confirm, demo, evaluate_budget, freeze

pytestmark = pytest.mark.requires_system_ffmpeg


@pytest.fixture(autouse=True)
def system_media_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    ffmpeg_executable = shutil.which("ffmpeg")
    ffprobe_executable = shutil.which("ffprobe")
    if ffmpeg_executable is None or ffprobe_executable is None:
        pytest.skip("the CPU example tests require installed FFmpeg and ffprobe")
    version_result = subprocess.run(
        [ffmpeg_executable, "-version"], capture_output=True, text=True, check=True, timeout=10
    )
    version_match = re.search(r"ffmpeg version n?(\d+)\.(\d+)", version_result.stdout)
    if version_match is not None and tuple(map(int, version_match.groups())) < (5, 1):
        pytest.skip("the CPU example tests require FFmpeg 5.1 or newer")
    monkeypatch.setenv("HFLOW_FFMPEG", ffmpeg_executable)
    monkeypatch.setenv("HFLOW_FFPROBE", ffprobe_executable)


def test_single_frame_budget_cannot_hide_unassessable_motion(tmp_path: Path) -> None:
    report = evaluate_budget(
        PreparationBudget(frames_per_second=1.0, width=32, height=24), tmp_path / "single-frame"
    )
    assert report["accuracy"] == 0.0
    assert report["episode_count"] == 8
    observations = json.loads((tmp_path / "single-frame" / "report.json").read_text())[
        "observations"
    ]
    assert all(observation["prediction"] is None for observation in observations)
    assert all(observation["frame_count"] == 1 for observation in observations)


def test_demo_freezes_a_cheaper_passing_budget_before_disjoint_confirmation(tmp_path: Path) -> None:
    output = tmp_path / "demo"
    demo(output)
    summary = json.loads((output / "summary.json").read_text())
    selected_index = summary["selected_trial"]
    assert summary["development_accuracies"][selected_index] == 1.0
    assert summary["pixel_frame_work"][selected_index] < summary["pixel_frame_work"][0]
    selected_report = json.loads((output / f"trial-{selected_index}" / "report.json").read_text())
    confirmation_report = json.loads((output / "confirmation" / "report.json").read_text())
    assert confirmation_report["phase"] == "confirmation"
    assert confirmation_report["accuracy"] == 1.0
    assert confirmation_report["budget"] == selected_report["budget"]
    assert {row["episode_id"] for row in selected_report["observations"]}.isdisjoint(
        row["episode_id"] for row in confirmation_report["observations"]
    )
    with pytest.raises(FileExistsError):
        demo(output)


def test_confirmation_rejects_modified_development_evidence_before_creating_output(
    tmp_path: Path,
) -> None:
    evaluate_budget(
        PreparationBudget(frames_per_second=8.0, width=64, height=48), tmp_path / "baseline"
    )
    report_path = tmp_path / "baseline" / "report.json"
    selection_path = tmp_path / "selection.json"
    freeze(report_path, selection_path)
    with report_path.open("a") as report_file:
        report_file.write("\n")
    confirmation_directory = tmp_path / "confirmation"
    with pytest.raises(ValueError, match="evidence changed"):
        confirm(selection_path, report_path, confirmation_directory)
    assert not confirmation_directory.exists()


def test_freeze_rejects_report_from_another_preparation_budget(tmp_path: Path) -> None:
    evaluate_budget(
        PreparationBudget(frames_per_second=1.0, width=32, height=24), tmp_path / "other-candidate"
    )
    selection_path = tmp_path / "selection.json"
    with pytest.raises(ValueError, match="does not match"):
        freeze(tmp_path / "other-candidate" / "report.json", selection_path)
    assert not selection_path.exists()
