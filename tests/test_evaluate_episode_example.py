import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.evaluate_episode import build_application


def test_recommended_episode_evaluation_combines_default_and_hosted_checks(
    tmp_path: Path,
) -> None:
    application = build_application(
        data_root=tmp_path,
        camera="/head_camera",
        frame_time_seconds=0.0,
    )

    assert [registered_check.name for registered_check in application.checks] == [
        "episode_duration",
        "timestamp_regularity",
        "camera_frame_stats",
        "keyframe_interval",
        "content_digest",
        "media_digest",
        "build_ai_hand_visibility",
        "build_ai_active_manipulation",
    ]
