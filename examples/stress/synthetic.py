#!/usr/bin/env python3
"""Synthetic stress corpus generator and ingest runner for hflow."""

import argparse
import hashlib
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from hflow.app import App
from hflow.checks import (
    action_rate,
    camera_frame_stats,
    episode_duration,
    idle_fraction,
    joint_discontinuity,
    timestamp_regularity,
)
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode
from hflow.transform import TransformConfig, write_canonical_episode

DEFAULT_EPISODES = 200
DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = "./stress_corpus"

CONVERTER_VERSION = "stress-synthetic-1.0"
JOINT_STATES_TOPIC = "/joint_states"


@dataclass(frozen=True)
class EpisodePlan:
    index: int
    duration_s: float
    num_cameras: int
    image_hz: float
    has_black_segment: bool
    black_segment: tuple[float, float] | None
    has_joint_jump: bool
    joint_jump_at_s: float | None


def _check_ffmpeg() -> None:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: ffmpeg not found. Install: sudo apt install ffmpeg", file=sys.stderr)
        sys.exit(1)


def _check_hflow_env() -> None:
    try:
        import importlib.util

        if importlib.util.find_spec("hflow") is None:
            raise ImportError
    except ImportError:
        print("ERROR: hflow not installed. Run `uv sync --locked --all-extras`", file=sys.stderr)
        sys.exit(1)


def _compute_pipeline_version() -> str:
    return hashlib.sha256(b"stress-synthetic-1.0").hexdigest()[:12]


def _plan_episodes(num_episodes: int, seed: int) -> list[EpisodePlan]:
    rng = random.Random(seed)
    plans: list[EpisodePlan] = []
    for i in range(num_episodes):
        duration = rng.uniform(2.0, 10.0)
        num_cameras = rng.randint(1, 3)
        img_hz = rng.choice([10.0, 15.0, 30.0])
        has_black = rng.random() < 0.2
        black_seg: tuple[float, float] | None = None
        if rng.random() < 0.2:
            start = rng.uniform(0.0, max(0.0, duration - 2.0))
            end = min(start + rng.uniform(0.5, 2.0), duration)
            black_seg = (start, end)
        has_jump = rng.random() < 0.2
        jump_at = rng.uniform(0.5, duration - 0.5) if rng.random() < 0.2 else None
        img_hz = rng.choice([10.0, 15.0, 30.0])
        plans.append(
            EpisodePlan(i, duration, num_cameras, img_hz, has_black, black_seg, has_jump, jump_at)
        )
    return plans


def generate_corpus(output_dir: Path, plans: list[EpisodePlan]) -> None:
    # Clean up existing episode files to ensure reproducibility
    for old_file in output_dir.glob("episode_*.mcap"):
        old_file.unlink(missing_ok=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    for plan in plans:
        spec = SyntheticEpisodeSpec(
            duration_s=plan.duration_s,
            cameras=tuple(f"cam_{j}" for j in range(plan.num_cameras)),
            image_hz=plan.image_hz,
            image_width=320,
            image_height=240,
            joint_hz=100.0,
            joint_count=7,
            black_segment=plan.black_segment,
            joint_jump_at_s=plan.joint_jump_at_s,
            joint_jump_rad=0.8,
            timestamp_offset_segment=None,
            start_time_ns=1_755_000_000_000_000_000 + plan.index * 1_000_000_000_000,
            task="stress_test",
            operator="synthetic",
            success=True,
            robot_software_version="stress-1.0",
            seed=plan.index,
        )
        with tempfile.NamedTemporaryFile(suffix=".mcap", delete=False) as tmp:
            input_mcap = Path(tmp.name)
        try:
            synthesize_episode(input_mcap, spec)
            output_path = output_dir / f"episode_{plan.index:06d}.mcap"
            write_canonical_episode(
                input_mcap,
                output_path,
                TransformConfig(),
                source_uri=f"stress://synthetic/ep_{plan.index:04d}",
            )
        finally:
            input_mcap.unlink(missing_ok=True)
        if (plan.index + 1) % 10 == 0:
            print(f"generated {plan.index + 1}/{len(plans)} episodes")


def run_ingest(corpus_dir: Path, data_root: Path) -> list[dict]:
    import hflow as hflow_module

    app = App("stress-test", data_root=data_root)
    app.check()(timestamp_regularity)
    app.check()(joint_discontinuity)
    app.check()(camera_frame_stats)
    app.check()(idle_fraction)
    app.check()(episode_duration)

    @app.check(name="action_rate_check")
    def action_rate_check(ep: hflow_module.Episode) -> hflow_module.CheckResult:
        return action_rate(ep, topics=["/joint_states"])

    mcap_files = sorted(corpus_dir.glob("episode_*.mcap"))
    print(f"Found {len(mcap_files)} episodes to ingest")

    results: list[dict] = []
    for i, mcap in enumerate(mcap_files):
        ep_start = time.perf_counter()
        ep_index = int(mcap.stem.split("_")[-1])
        print(f"Processing {i + 1}/{len(mcap_files)}: {mcap.name}")
        try:
            report = app.process(str(mcap), record=True)
            ep_duration = time.perf_counter() - ep_start
            check_statuses = {run.check.name: run.status.name for run in report.checks}
            results.append(
                {
                    "episode_index": ep_index,
                    "duration_s": ep_duration,
                    "status": "OK",
                    "check_statuses": dict(check_statuses),
                }
            )
            print(f"  OK: {ep_duration:.2f}s")
        except Exception as e:
            ep_duration = time.perf_counter() - ep_start
            results.append(
                {
                    "episode_index": ep_index,
                    "duration_s": ep_duration,
                    "status": "ERROR",
                    "error": str(e),
                }
            )
            print(f"  FAILED: {e}")
    return results


def print_summary(results: list[dict], ingest_time: float) -> None:
    ok = sum(1 for r in results if r.get("status") == "OK")
    err = sum(1 for r in results if r.get("status") == "ERROR")
    total = len(results)
    avg_ms = sum(r["duration_s"] for r in results) / total * 1000 if total else 0
    status_counts: dict[str, int] = {}
    for r in results:
        for s in r.get("check_statuses", {}).values():
            status_counts[s] = status_counts.get(s, 0) + 1

    print("\n" + "=" * 60)
    print("STRESS TEST SUMMARY")
    print("=" * 60)
    print(f"Episodes: {len(results)} | OK: {ok} | ERROR: {err}")
    print(f"Total time: {ingest_time:.1f}s | Avg/ep: {avg_ms:.0f}ms")
    print("\nCheck status distribution:")
    for k, v in sorted(status_counts.items()):
        print(f"  {k}: {v}")
    if any(r.get("status") == "ERROR" for r in results):
        print("\nErrors:")
        for r in results:
            if r.get("status") == "ERROR":
                print(f"  ep_{r['episode_index']:04d}: {r.get('error')}")
    print("=" * 60)


def _check_ffmpeg() -> None:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: ffmpeg not found. Install: sudo apt install ffmpeg", file=sys.stderr)
        sys.exit(1)


def _check_hflow_env() -> None:
    try:
        import importlib.util

        if importlib.util.find_spec("hflow") is None:
            raise ImportError
    except ImportError:
        print("ERROR: hflow not installed. Run `uv sync --locked --all-extras`", file=sys.stderr)
        sys.exit(1)


def _compute_pipeline_version() -> str:
    return hashlib.sha256(b"stress-synthetic-1.0").hexdigest()[:12]


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Synthetic stress corpus generator + ingest for hflow",
    )
    parser.add_argument("--output-dir", type=Path, default="./stress_corpus")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--ingest-only", action="store_true")
    parser.add_argument("--data-root", type=Path, default="./data")
    args = parser.parse_args()

    if args.generate_only and args.ingest_only:
        print("ERROR: --generate-only and --ingest-only are mutually exclusive", file=sys.stderr)
        sys.exit(1)

    corpus_dir = args.output_dir
    _check_ffmpeg()
    _check_hflow_env()

    if not args.ingest_only:
        print(f"Generating {args.episodes} episodes in {corpus_dir} (seed={args.seed})")
        gen_start = time.perf_counter()
        plans = _plan_episodes(args.episodes, args.seed)
        generate_corpus(args.output_dir, plans)
        print(f"\nGeneration complete in {time.perf_counter() - gen_start:.1f}s")

    if not args.generate_only:
        corpus_dir = args.output_dir
        if not corpus_dir.exists():
            print(f"ERROR: corpus directory {corpus_dir} does not exist", file=sys.stderr)
            sys.exit(1)
        print(f"\nIngesting episodes from {corpus_dir}")
        ingest_start = time.perf_counter()
        results = run_ingest(corpus_dir, args.data_root)
        ingest_time = time.perf_counter() - ingest_start
        print_summary(results, ingest_time)


if __name__ == "__main__":
    main()
