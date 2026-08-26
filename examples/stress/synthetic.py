#!/usr/bin/env python3
"""Generate a seeded synthetic corpus and run the local ingest path over it.

The point is a repeatable number. A given ``--episodes``/``--seed`` pair always
plans the same corpus, so a run today and a run after a change are directly
comparable. Nothing here is a fixture: the corpus lands under ``--output-dir``
and the catalog under ``--data-root``, both of which are ignored trees.
"""

import argparse
import random
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import hflow
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
DEFAULT_OUTPUT_DIR = Path("./stress_corpus")
DEFAULT_DATA_ROOT = Path("./data")

JOINT_STATES_TOPIC = "/joint_states"
FAULT_PROBABILITY = 0.2
IMAGE_RATES_HZ = (10.0, 15.0, 30.0)
EPISODE_START_TIME_NS = 1_755_000_000_000_000_000


@dataclass(frozen=True)
class EpisodePlan:
    """One episode's shape, drawn from the seeded planner."""

    index: int
    duration_s: float
    num_cameras: int
    image_hz: float
    black_segment: tuple[float, float] | None
    joint_jump_at_s: float | None


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        print(
            "ERROR: ffmpeg not found on PATH. Install it (for example "
            "`sudo apt install ffmpeg`) and run again.",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f"ERROR: ffmpeg is on PATH but not runnable: {exc}", file=sys.stderr)
        sys.exit(1)


def _plan_episodes(num_episodes: int, seed: int) -> list[EpisodePlan]:
    """Draw the corpus plan. Same arguments in, same plan out."""
    rng = random.Random(seed)
    plans: list[EpisodePlan] = []
    for index in range(num_episodes):
        duration_s = rng.uniform(2.0, 10.0)
        num_cameras = rng.randint(1, 3)
        image_hz = rng.choice(IMAGE_RATES_HZ)
        black_segment: tuple[float, float] | None = None
        if rng.random() < FAULT_PROBABILITY:
            start = rng.uniform(0.0, max(0.0, duration_s - 2.0))
            black_segment = (start, min(start + rng.uniform(0.5, 2.0), duration_s))
        joint_jump_at_s = (
            rng.uniform(0.5, duration_s - 0.5) if rng.random() < FAULT_PROBABILITY else None
        )
        plans.append(
            EpisodePlan(
                index=index,
                duration_s=duration_s,
                num_cameras=num_cameras,
                image_hz=image_hz,
                black_segment=black_segment,
                joint_jump_at_s=joint_jump_at_s,
            )
        )
    return plans


def generate_corpus(output_dir: Path, plans: list[EpisodePlan]) -> None:
    """Write one canonical episode per plan, replacing any previous corpus."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_episode in output_dir.glob("episode_*.mcap"):
        stale_episode.unlink(missing_ok=True)

    for plan in plans:
        spec = SyntheticEpisodeSpec(
            duration_s=plan.duration_s,
            cameras=tuple(f"cam_{camera}" for camera in range(plan.num_cameras)),
            image_hz=plan.image_hz,
            image_width=320,
            image_height=240,
            joint_hz=100.0,
            joint_count=7,
            black_segment=plan.black_segment,
            joint_jump_at_s=plan.joint_jump_at_s,
            joint_jump_rad=0.8,
            timestamp_offset_segment=None,
            start_time_ns=EPISODE_START_TIME_NS + plan.index * 1_000_000_000_000,
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
            write_canonical_episode(
                input_mcap,
                output_dir / f"episode_{plan.index:06d}.mcap",
                TransformConfig(),
                source_uri=f"stress://synthetic/ep_{plan.index:04d}",
            )
        finally:
            input_mcap.unlink(missing_ok=True)
        if (plan.index + 1) % 10 == 0:
            print(f"generated {plan.index + 1}/{len(plans)} episodes")


def run_ingest(corpus_dir: Path, data_root: Path) -> list[dict]:
    """Process every episode in the corpus, timing each one."""
    app = hflow.App("stress-test", data_root=data_root)
    app.check(version="1")(timestamp_regularity)
    app.check(version="1")(joint_discontinuity)
    app.check(version="1")(camera_frame_stats)
    app.check(version="1")(idle_fraction)
    app.check(version="1")(episode_duration)

    @app.check(version="1", name="action_rate_check")
    def action_rate_check(episode: hflow.Episode) -> hflow.CheckResult:
        return action_rate(episode, topics=[JOINT_STATES_TOPIC])

    mcap_files = sorted(corpus_dir.glob("episode_*.mcap"))
    print(f"Found {len(mcap_files)} episodes to ingest")

    results: list[dict] = []
    for position, mcap in enumerate(mcap_files, start=1):
        episode_index = int(mcap.stem.split("_")[-1])
        print(f"Processing {position}/{len(mcap_files)}: {mcap.name}")
        started = time.perf_counter()
        try:
            report = app.process(str(mcap), record=True)
        except Exception as exc:
            elapsed_s = time.perf_counter() - started
            results.append(
                {
                    "episode_index": episode_index,
                    "duration_s": elapsed_s,
                    "status": "ERROR",
                    "error": str(exc),
                    "check_statuses": {},
                }
            )
            print(f"  FAILED: {exc}")
            continue
        elapsed_s = time.perf_counter() - started
        results.append(
            {
                "episode_index": episode_index,
                "duration_s": elapsed_s,
                "status": "OK",
                "check_statuses": {run.check.name: run.status.name for run in report.checks},
            }
        )
        print(f"  OK: {elapsed_s:.2f}s")
    return results


def print_summary(results: list[dict], ingest_time_s: float) -> None:
    """Print the per-run totals and the check status distribution."""
    total = len(results)
    ok = sum(1 for result in results if result["status"] == "OK")
    errored = [result for result in results if result["status"] == "ERROR"]
    average_ms = sum(result["duration_s"] for result in results) / total * 1000 if total else 0.0
    status_counts: dict[str, int] = {}
    for result in results:
        for status in result["check_statuses"].values():
            status_counts[status] = status_counts.get(status, 0) + 1

    print("\n" + "=" * 60)
    print("STRESS TEST SUMMARY")
    print("=" * 60)
    print(f"Episodes: {total} | OK: {ok} | ERROR: {len(errored)}")
    print(f"Total time: {ingest_time_s:.1f}s | Avg/ep: {average_ms:.0f}ms")
    print("\nCheck status distribution:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    if errored:
        print("\nErrors:")
        for result in errored:
            print(f"  ep_{result['episode_index']:04d}: {result['error']}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Generate a seeded synthetic corpus and run the local ingest path over it.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--ingest-only", action="store_true")
    args = parser.parse_args()

    if args.generate_only and args.ingest_only:
        print("ERROR: --generate-only and --ingest-only are mutually exclusive", file=sys.stderr)
        sys.exit(1)

    _require_ffmpeg()

    if not args.ingest_only:
        print(f"Generating {args.episodes} episodes in {args.output_dir} (seed={args.seed})")
        generation_started = time.perf_counter()
        generate_corpus(args.output_dir, _plan_episodes(args.episodes, args.seed))
        print(f"\nGeneration complete in {time.perf_counter() - generation_started:.1f}s")

    if not args.generate_only:
        if not args.output_dir.exists():
            print(f"ERROR: corpus directory {args.output_dir} does not exist", file=sys.stderr)
            sys.exit(1)
        print(f"\nIngesting episodes from {args.output_dir}")
        ingest_started = time.perf_counter()
        results = run_ingest(args.output_dir, args.data_root)
        print_summary(results, time.perf_counter() - ingest_started)


if __name__ == "__main__":
    main()
