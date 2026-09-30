"""A frozen CPU evaluator for data-preparation experiments, outside HFlow core.

Synthetic marker motion exercises temporal sampling. It does not estimate VLM,
robot-policy, or real-world perception quality. Each evaluation uses a fresh
output directory, real HFlow encoding, and a fixed color-based vision function.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Literal

import av
import numpy as np
import typer
from hflow.ffmpeg import ffmpeg_version
from hflow.importers import VideoImportConfig, prepare_model_video
from hflow.transform import TransformConfig
from pydantic import BaseModel, ConfigDict, Field

from candidate import preparation_configuration

app = typer.Typer(help=__doc__)
DEVELOPMENT_SEEDS = tuple(range(8))
CONFIRMATION_SEEDS = tuple(range(100, 108))
ENCODING_SETTINGS = TransformConfig(crf=18)


class PreparationBudget(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)
    frames_per_second: Annotated[float, Field(ge=1, le=16)]
    width: Annotated[int, Field(ge=16, le=192, multiple_of=2)]
    height: Annotated[int, Field(ge=16, le=128, multiple_of=2)]

    def configuration(self) -> VideoImportConfig:
        return VideoImportConfig(
            duration_s=1.0,
            image_hz=self.frames_per_second,
            image_width=self.width,
            image_height=self.height,
        )


class FrozenSelection(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)
    schema_version: Literal[1] = 1
    evaluator_sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    development_report_sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    budget: PreparationBudget


def sha256_file(path: Path) -> str:
    with path.open("rb") as input_file:
        return hashlib.file_digest(input_file, "sha256").hexdigest()


def write_new_json(path: Path, value: object) -> None:
    serialized = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x") as output_file:
        output_file.write(serialized)


def read_json_object(path: Path) -> dict[str, object]:
    def unique_fields(fields: list[tuple[str, object]]) -> dict[str, object]:
        result = dict(fields)
        if len(result) != len(fields):
            raise ValueError("duplicate evidence fields")
        return result

    value = json.loads(path.read_text(), object_pairs_hook=unique_fields)
    if not isinstance(value, dict):
        raise ValueError("evidence must be a JSON object")
    return value


def synthesize_motion_video(output: Path, seed: int) -> Literal["left", "right"]:
    """One independently generated episode with a marker that begins moving late."""
    generator = random.Random(seed)
    moves_right = seed % 2 == 0
    motion_start = generator.uniform(0.35, 0.7)
    initial_column, final_column = (10, 78) if moves_right else (78, 10)
    marker_row = generator.randrange(10, 40)
    with av.open(str(output), "w") as container:
        stream = container.add_stream("libx264", rate=16)
        stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
        stream.options = {"crf": "18", "threads": "1"}
        for frame_index in range(16):
            progress = max(0.0, (frame_index / 16 - motion_start) / (1 - motion_start))
            marker_column = round(initial_column + (final_column - initial_column) * progress)
            pixels = np.full((64, 96, 3), generator.randrange(15, 35), dtype=np.uint8)
            pixels[marker_row : marker_row + 8, marker_column : marker_column + 8] = (240, 20, 20)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return "right" if moves_right else "left"


def measure_motion(video: Path) -> tuple[str | None, int]:
    """Fixed visual baseline; unknown observations still count against accuracy."""
    marker_positions: list[float] = []
    decoded_frame_count = 0
    with av.open(str(video)) as container:
        for frame in container.decode(video=0):
            decoded_frame_count += 1
            pixels = frame.to_ndarray(format="rgb24").astype(np.int16)
            marker_mask = (pixels[:, :, 0] > 160) & (pixels[:, :, 0] - pixels[:, :, 1] > 80)
            _, columns = np.nonzero(marker_mask)
            if columns.size:
                marker_positions.append(float(columns.mean()) / frame.width)
    if len(marker_positions) < 2:
        return None, decoded_frame_count
    displacement = marker_positions[-1] - marker_positions[0]
    if abs(displacement) < 0.03:
        return None, decoded_frame_count
    return ("right" if displacement > 0 else "left"), decoded_frame_count


def evaluate_budget(
    budget: PreparationBudget,
    output: Path,
    *,
    phase: Literal["development", "confirmation"] = "development",
) -> dict[str, object]:
    """Prepare and measure real media; candidate failures cannot disappear from scores."""
    output.mkdir(parents=True, exist_ok=False)
    configuration = budget.configuration()
    seeds = DEVELOPMENT_SEEDS if phase == "development" else CONFIRMATION_SEEDS
    observations: list[dict[str, object]] = []
    correct_count = 0
    pixel_frame_work = 0
    preparation_seconds = 0.0
    for seed in seeds:
        source = output / f"source-{seed}.mp4"
        target = synthesize_motion_video(source, seed)
        prepared_path = output / f"prepared-{seed}.mp4"
        preparation_started = time.perf_counter()
        outcome = prepare_model_video(
            source, prepared_path, configuration, transform_config=ENCODING_SETTINGS
        )
        elapsed_seconds = time.perf_counter() - preparation_started
        preparation_seconds += elapsed_seconds
        prediction, frame_count = (
            measure_motion(outcome) if isinstance(outcome, Path) else (None, 0)
        )
        if isinstance(outcome, Path) and frame_count != configuration.frame_count:
            raise RuntimeError("prepared video violates the requested sample count")
        correct_count += prediction == target
        pixel_frame_work += frame_count * configuration.image_width * configuration.image_height
        observations.append(
            {
                "episode_id": f"synthetic-motion-{seed}",
                "source_sha256": sha256_file(source),
                "prepared_sha256": sha256_file(outcome) if isinstance(outcome, Path) else None,
                "status": "prepared" if isinstance(outcome, Path) else type(outcome).__name__,
                "target": target,
                "prediction": prediction,
                "frame_count": frame_count,
                "preparation_seconds": elapsed_seconds,
            }
        )
    report: dict[str, object] = {
        "schema_version": 1,
        "phase": phase,
        "scope": "synthetic-cpu-preparation-smoke",
        "hflow_version": version("hflow"),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "av_version": av.__version__,
        "ffmpeg_version": ffmpeg_version(),
        "evaluator_sha256": sha256_file(Path(__file__)),
        "budget": budget.model_dump(),
        "encoding_settings": asdict(ENCODING_SETTINGS),
        "episode_count": len(seeds),
        "correct_count": correct_count,
        "accuracy": correct_count / len(seeds),
        "pixel_frame_work": pixel_frame_work,
        "preparation_seconds": preparation_seconds,
        "observations": observations,
    }
    write_new_json(output / "report.json", report)
    return report


def candidate_budget() -> PreparationBudget:
    configuration = preparation_configuration()
    # All other input and encoding settings belong to this frozen evaluator.
    budget = PreparationBudget(
        frames_per_second=configuration.image_hz,
        width=configuration.image_width,
        height=configuration.image_height,
    )
    if configuration != budget.configuration():
        raise ValueError("candidate may change only frame rate, width and height")
    return budget


@app.command()
def evaluate(output: Path = typer.Option(...)) -> None:
    report = evaluate_budget(candidate_budget(), output)
    print(json.dumps(report, indent=2))


@app.command()
def freeze(development_report: Path, output: Path = typer.Option(...)) -> None:
    """Lock the current candidate only after reproducing a development result."""
    report = read_json_object(development_report)
    budget = candidate_budget()
    evaluator_hash = sha256_file(Path(__file__))
    if (
        report.get("phase") != "development"
        or report.get("budget") != budget.model_dump()
        or report.get("evaluator_sha256") != evaluator_hash
    ):
        raise ValueError("development report does not match the current candidate and evaluator")
    if report.get("accuracy") != 1.0:
        raise ValueError("this smoke example requires every development episode to pass")
    selection = FrozenSelection(
        evaluator_sha256=evaluator_hash,
        development_report_sha256=sha256_file(development_report),
        budget=budget,
    )
    write_new_json(output, selection.model_dump())


@app.command()
def confirm(
    selection: Path,
    development_report: Path,
    output: Path = typer.Option(...),
) -> None:
    frozen = FrozenSelection.model_validate(read_json_object(selection))
    if frozen.evaluator_sha256 != sha256_file(
        Path(__file__)
    ) or frozen.development_report_sha256 != sha256_file(development_report):
        raise ValueError("selected development evidence changed")
    report = evaluate_budget(frozen.budget, output, phase="confirmation")
    print(json.dumps(report, indent=2))


@app.command()
def demo(output: Path = typer.Option(...)) -> None:
    """Run a bounded CPU sweep to check the example before invoking an agent."""
    output.mkdir(parents=True, exist_ok=False)
    budgets = [
        PreparationBudget(frames_per_second=8.0, width=64, height=48),
        PreparationBudget(frames_per_second=1.0, width=32, height=24),
        PreparationBudget(frames_per_second=2.0, width=32, height=24),
        PreparationBudget(frames_per_second=4.0, width=32, height=24),
    ]
    reports = [
        evaluate_budget(budget, output / f"trial-{trial_index}")
        for trial_index, budget in enumerate(budgets)
    ]
    passing_indices = [index for index, report in enumerate(reports) if report["accuracy"] == 1.0]
    if not passing_indices:
        raise RuntimeError("no preparation budget met the development criterion")
    selected_index = min(
        passing_indices,
        key=lambda index: (
            budgets[index].configuration().frame_count
            * budgets[index].width
            * budgets[index].height
        ),
    )
    selected_report = output / f"trial-{selected_index}" / "report.json"
    selection = FrozenSelection(
        evaluator_sha256=sha256_file(Path(__file__)),
        development_report_sha256=sha256_file(selected_report),
        budget=budgets[selected_index],
    )
    selection_path = output / "selection.json"
    write_new_json(selection_path, selection.model_dump())
    confirm(selection_path, selected_report, output / "confirmation")
    summary = {
        "scope": "synthetic-cpu-preparation-smoke",
        "selected_trial": selected_index,
        "selected_budget": selection.budget.model_dump(),
        "development_accuracies": [report["accuracy"] for report in reports],
        "pixel_frame_work": [report["pixel_frame_work"] for report in reports],
        "confirmation_report": "confirmation/report.json",
    }
    write_new_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()
