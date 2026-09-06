"""Camera-shake measurement: it must tell a shaking camera from a moving scene.

Fixtures crop a moving window out of one STILL textured image, so the scene is
byte-identical across cases and only the camera path differs. Using an animated
source instead would make scene motion indistinguishable from camera motion --
which is the whole question the check exists to answer, so a fixture that blurs
it proves nothing.
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest

import hflow
from hflow._video_measurement_toolchain import resolved_video_measurement_toolchain
from hflow._video_measurements import (
    CameraMotionMeasurements,
    CameraMotionSettings,
    InsufficientVideoFrames,
    measure_camera_motion,
)
from hflow.checks import camera_stability
from hflow.ffmpeg import ffmpeg_path

pytest.importorskip("cv2", reason="camera-motion measurement needs the 'motion' extra")

_FRAMES_PER_SECOND = 30
_DURATION_S = 2
_SHAKE_HZ = 8


@pytest.fixture(scope="module")
def still_texture(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One frame of dense static texture, large enough to crop a window from."""
    path = tmp_path_factory.mktemp("motion") / "still.png"
    subprocess.run(
        [
            str(ffmpeg_path()),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x480:rate=1:duration=1",
            "-frames:v",
            "1",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def _render_camera_path(still_texture: Path, output: Path, offset_x: str, offset_y: str) -> Path:
    subprocess.run(
        [
            str(ffmpeg_path()),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-loop",
            "1",
            "-framerate",
            str(_FRAMES_PER_SECOND),
            "-t",
            str(_DURATION_S),
            "-i",
            str(still_texture),
            "-vf",
            f"crop=320:240:{offset_x}:{offset_y}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-x264-params",
            "keyint=30:min-keyint=30:scenecut=0:bframes=0",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    return output


def _shake(amplitude_px: int) -> tuple[str, str]:
    return (
        f"160+{amplitude_px}*sin(2*PI*{_SHAKE_HZ}*t)",
        f"120+{amplitude_px}*cos(2*PI*{_SHAKE_HZ}*t)",
    )


def _measure(video: Path) -> CameraMotionMeasurements:
    camera_motion_result = measure_camera_motion(
        video,
        toolchain=resolved_video_measurement_toolchain(),
        settings=CameraMotionSettings(frames_per_second=_FRAMES_PER_SECOND),
    )
    assert isinstance(camera_motion_result, CameraMotionMeasurements), (
        "the fixture must contain an adjacent frame pair"
    )
    return camera_motion_result


def test_a_single_frame_is_an_explicit_recoverable_result(
    still_texture: Path,
) -> None:
    camera_motion_result = measure_camera_motion(
        still_texture,
        toolchain=resolved_video_measurement_toolchain(),
        settings=CameraMotionSettings(frames_per_second=_FRAMES_PER_SECOND),
    )
    assert isinstance(camera_motion_result, InsufficientVideoFrames)
    assert camera_motion_result.observed_frame_count == 1
    assert camera_motion_result.minimum_required_frame_count == 2


def test_a_static_camera_reports_no_unstable_footage(still_texture: Path, tmp_path: Path) -> None:
    motion = _measure(_render_camera_path(still_texture, tmp_path / "static.mp4", "160", "120"))
    assert motion.unstable_share == 0.0
    # Well under the instrument's own resolution: nothing moved.
    assert motion.shake_rate_p50_degrees_per_second < motion.resolution_floor_degrees_per_second


def test_a_smooth_pan_is_not_reported_as_unstable(still_texture: Path, tmp_path: Path) -> None:
    """The discrimination that frame differencing cannot make: a deliberate pan
    moves every pixel a great deal and is not shake.
    """
    motion = _measure(_render_camera_path(still_texture, tmp_path / "pan.mp4", "160+60*t", "120"))
    assert motion.unstable_share == 0.0


def test_measured_shake_is_linear_in_its_amplitude(still_texture: Path, tmp_path: Path) -> None:
    """The property that makes the number mean something: twice the shake reads
    twice as shaky. Without it the measurement could only rank footage, and a
    rate in degrees per second would be a label rather than a quantity.
    """
    amplitudes_px = (2, 8, 24)
    rates_per_pixel = []
    for amplitude_px in amplitudes_px:
        motion = _measure(
            _render_camera_path(
                still_texture, tmp_path / f"shake{amplitude_px}.mp4", *_shake(amplitude_px)
            )
        )
        assert motion.unstable_share > 0.5, f"{amplitude_px}px of wobble is shake"
        rates_per_pixel.append(motion.shake_rate_p50_degrees_per_second / amplitude_px)

    # Across a 12x range of amplitude the rate per pixel must stay put.
    assert max(rates_per_pixel) / min(rates_per_pixel) < 1.3, rates_per_pixel


def test_shake_below_the_resolution_floor_is_not_called_unstable(
    still_texture: Path, tmp_path: Path
) -> None:
    """The instrument states its own resolution and refuses to report below it,
    rather than dressing sub-pixel noise up as a defect.
    """
    motion = _measure(
        _render_camera_path(
            still_texture,
            tmp_path / "subpixel.mp4",
            f"160+0.25*sin(2*PI*{_SHAKE_HZ}*t)",
            f"120+0.25*cos(2*PI*{_SHAKE_HZ}*t)",
        )
    )
    assert motion.resolution_floor_degrees_per_second > 0.0
    assert motion.shake_rate_p50_degrees_per_second < motion.resolution_floor_degrees_per_second
    assert motion.unstable_share == 0.0


def test_every_pair_is_accounted_for_as_measured_or_unclassified(
    still_texture: Path, tmp_path: Path
) -> None:
    """A share over an unstated denominator is not a measurement."""
    motion = _measure(_render_camera_path(still_texture, tmp_path / "pan.mp4", "160+30*t", "120"))
    pair_count = _FRAMES_PER_SECOND * _DURATION_S - 1
    accounted_s = motion.measured_seconds + motion.unclassified_seconds
    assert accounted_s == pytest.approx(pair_count / _FRAMES_PER_SECOND, abs=0.05)
    assert motion.unstable_seconds <= motion.measured_seconds


def test_footage_with_nothing_to_track_reports_no_coverage_not_steadiness(
    tmp_path: Path,
) -> None:
    """Featureless footage cannot be measured, and must say so.

    The dangerous failure here is reporting it as steady: a caller reading
    ``unstable_share`` alone would see 0.0 and conclude the camera was fine,
    when in fact no transform could be fitted to a single pair. Reporting the
    whole clip as unclassified, with coverage at zero, is what makes the share
    honest -- and this is the only fixture that drives the unmeasurable side of
    the accounting above zero at all.
    """
    flat = tmp_path / "flat.mp4"
    subprocess.run(
        [
            str(ffmpeg_path()),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=gray:size=320x240:rate={_FRAMES_PER_SECOND}:duration={_DURATION_S}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-x264-params",
            "keyint=30:min-keyint=30:scenecut=0:bframes=0",
            str(flat),
        ],
        check=True,
        capture_output=True,
    )
    motion = _measure(flat)

    pair_count = _FRAMES_PER_SECOND * _DURATION_S - 1
    assert motion.measured_seconds == 0.0
    assert motion.unclassified_seconds == pytest.approx(pair_count / _FRAMES_PER_SECOND, abs=0.05)
    assert motion.unstable_seconds == 0.0
    # Zero over nothing measured, which the coverage beside it qualifies.
    assert motion.unstable_share == 0.0

    # The check turns that into a coverage of zero rather than a clean bill.
    from hflow.testing import VideoEpisodeSpec, write_video_episode
    from hflow.transform import TransformConfig, write_canonical_episode

    source = write_video_episode(
        flat,
        tmp_path / "episode.mcap",
        VideoEpisodeSpec(
            duration_s=float(_DURATION_S),
            image_hz=float(_FRAMES_PER_SECOND),
            image_width=320,
            image_height=240,
            camera_name="head_camera",
        ),
    )
    canonical = tmp_path / "episode.canonical.mcap"
    write_canonical_episode(source, canonical, TransformConfig())
    with hflow.Episode(canonical) as episode:
        camera_topic = episode.cameras[0]
        result = camera_stability(episode)
    assert result.measurements[f"{camera_topic}/coverage_share"] == 0.0
    assert result.intervals == []


def test_the_check_reports_stability_with_its_coverage(still_texture: Path, tmp_path: Path) -> None:
    """End to end through a canonical episode, which is what a pipeline sees."""
    from hflow.testing import VideoEpisodeSpec, write_video_episode
    from hflow.transform import TransformConfig, write_canonical_episode

    shaky = _render_camera_path(still_texture, tmp_path / "shaky.mp4", *_shake(24))
    source = write_video_episode(
        shaky,
        tmp_path / "episode.mcap",
        VideoEpisodeSpec(
            duration_s=float(_DURATION_S),
            image_hz=float(_FRAMES_PER_SECOND),
            image_width=320,
            image_height=240,
            camera_name="head_camera",
        ),
    )
    canonical = tmp_path / "episode.canonical.mcap"
    write_canonical_episode(source, canonical, TransformConfig())
    with hflow.Episode(canonical) as episode:
        camera_topic = episode.cameras[0]
        result = camera_stability(episode)

    unstable_share = result.measurements[f"{camera_topic}/unstable_share"]
    coverage_share = result.measurements[f"{camera_topic}/coverage_share"]
    assert isinstance(unstable_share, float) and unstable_share > 0.5
    assert isinstance(coverage_share, float) and coverage_share > 0.9
    assert result.measurements[f"{camera_topic}/horizontal_fov_degrees"] == 90.0
    assert [i.label for i in result.intervals] == [f"unstable:{camera_topic}"] * len(
        result.intervals
    )
    assert result.intervals, "shaky footage must localize somewhere"
    assert result.verdict is None


def test_camera_stability_rejects_negative_shake_threshold(
    still_texture: Path, tmp_path: Path
) -> None:
    """A negative shake threshold is rejected before camera measurement."""
    from hflow.testing import VideoEpisodeSpec, write_video_episode
    from hflow.transform import TransformConfig, write_canonical_episode

    shaky = _render_camera_path(still_texture, tmp_path / "shaky.mp4", *_shake(24))
    source = write_video_episode(
        shaky,
        tmp_path / "episode.mcap",
        VideoEpisodeSpec(
            duration_s=float(_DURATION_S),
            image_hz=float(_FRAMES_PER_SECOND),
            image_width=320,
            image_height=240,
            camera_name="head_camera",
        ),
    )
    canonical = tmp_path / "episode.canonical.mcap"
    write_canonical_episode(source, canonical, TransformConfig())

    with (
        hflow.Episode(canonical) as episode,
        pytest.raises(
            ValueError,
            match=r"^shake_threshold_dps must be finite and non-negative$",
        ),
    ):
        camera_stability(episode, shake_threshold_dps=-1.0)


def test_camera_stability_rejects_nan_shake_threshold(still_texture: Path, tmp_path: Path) -> None:
    """A NaN shake threshold is rejected before camera measurement."""
    from hflow.testing import VideoEpisodeSpec, write_video_episode
    from hflow.transform import TransformConfig, write_canonical_episode

    shaky = _render_camera_path(still_texture, tmp_path / "shaky.mp4", *_shake(24))
    source = write_video_episode(
        shaky,
        tmp_path / "episode.mcap",
        VideoEpisodeSpec(
            duration_s=float(_DURATION_S),
            image_hz=float(_FRAMES_PER_SECOND),
            image_width=320,
            image_height=240,
            camera_name="head_camera",
        ),
    )
    canonical = tmp_path / "episode.canonical.mcap"
    write_canonical_episode(source, canonical, TransformConfig())

    with (
        hflow.Episode(canonical) as episode,
        pytest.raises(
            ValueError,
            match=r"^shake_threshold_dps must be finite and non-negative$",
        ),
    ):
        camera_stability(episode, shake_threshold_dps=np.nan)


def test_camera_stability_rejects_infinite_shake_threshold(
    still_texture: Path, tmp_path: Path
) -> None:
    """An infinite shake threshold is rejected before camera measurement."""
    from hflow.testing import VideoEpisodeSpec, write_video_episode
    from hflow.transform import TransformConfig, write_canonical_episode

    shaky = _render_camera_path(still_texture, tmp_path / "shaky.mp4", *_shake(24))
    source = write_video_episode(
        shaky,
        tmp_path / "episode.mcap",
        VideoEpisodeSpec(
            duration_s=float(_DURATION_S),
            image_hz=float(_FRAMES_PER_SECOND),
            image_width=320,
            image_height=240,
            camera_name="head_camera",
        ),
    )
    canonical = tmp_path / "episode.canonical.mcap"
    write_canonical_episode(source, canonical, TransformConfig())

    with (
        hflow.Episode(canonical) as episode,
        pytest.raises(
            ValueError,
            match=r"^shake_threshold_dps must be finite and non-negative$",
        ),
    ):
        camera_stability(episode, shake_threshold_dps=np.inf)


def test_camera_stability_rejects_bool_shake_threshold(still_texture: Path, tmp_path: Path) -> None:
    """A boolean shake threshold is rejected instead of being treated as 1."""
    from hflow.testing import VideoEpisodeSpec, write_video_episode
    from hflow.transform import TransformConfig, write_canonical_episode

    shaky = _render_camera_path(still_texture, tmp_path / "shaky.mp4", *_shake(24))
    source = write_video_episode(
        shaky,
        tmp_path / "episode.mcap",
        VideoEpisodeSpec(
            duration_s=float(_DURATION_S),
            image_hz=float(_FRAMES_PER_SECOND),
            image_width=320,
            image_height=240,
            camera_name="head_camera",
        ),
    )
    canonical = tmp_path / "episode.canonical.mcap"
    write_canonical_episode(source, canonical, TransformConfig())

    with (
        hflow.Episode(canonical) as episode,
        pytest.raises(
            ValueError,
            match=r"^shake_threshold_dps must be finite and non-negative$",
        ),
    ):
        camera_stability(episode, shake_threshold_dps=True)


def test_camera_stability_rejects_negative_min_duration(
    still_texture: Path, tmp_path: Path
) -> None:
    """A negative minimum unstable duration is rejected before camera measurement."""
    from hflow.testing import VideoEpisodeSpec, write_video_episode
    from hflow.transform import TransformConfig, write_canonical_episode

    shaky = _render_camera_path(still_texture, tmp_path / "shaky.mp4", *_shake(24))
    source = write_video_episode(
        shaky,
        tmp_path / "episode.mcap",
        VideoEpisodeSpec(
            duration_s=float(_DURATION_S),
            image_hz=float(_FRAMES_PER_SECOND),
            image_width=320,
            image_height=240,
            camera_name="head_camera",
        ),
    )
    canonical = tmp_path / "episode.canonical.mcap"
    write_canonical_episode(source, canonical, TransformConfig())

    with (
        hflow.Episode(canonical) as episode,
        pytest.raises(
            ValueError,
            match=r"^unstable_min_duration_s must be finite and non-negative$",
        ),
    ):
        camera_stability(episode, unstable_min_duration_s=-1.0)


def test_camera_stability_rejects_nan_min_duration(still_texture: Path, tmp_path: Path) -> None:
    """A NaN minimum unstable duration is rejected before camera measurement."""
    from hflow.testing import VideoEpisodeSpec, write_video_episode
    from hflow.transform import TransformConfig, write_canonical_episode

    shaky = _render_camera_path(still_texture, tmp_path / "shaky.mp4", *_shake(24))
    source = write_video_episode(
        shaky,
        tmp_path / "episode.mcap",
        VideoEpisodeSpec(
            duration_s=float(_DURATION_S),
            image_hz=float(_FRAMES_PER_SECOND),
            image_width=320,
            image_height=240,
            camera_name="head_camera",
        ),
    )
    canonical = tmp_path / "episode.canonical.mcap"
    write_canonical_episode(source, canonical, TransformConfig())

    with (
        hflow.Episode(canonical) as episode,
        pytest.raises(
            ValueError,
            match=r"^unstable_min_duration_s must be finite and non-negative$",
        ),
    ):
        camera_stability(episode, unstable_min_duration_s=np.nan)


def test_camera_stability_rejects_infinite_min_duration(
    still_texture: Path, tmp_path: Path
) -> None:
    """An infinite minimum unstable duration is rejected before camera measurement."""
    from hflow.testing import VideoEpisodeSpec, write_video_episode
    from hflow.transform import TransformConfig, write_canonical_episode

    shaky = _render_camera_path(still_texture, tmp_path / "shaky.mp4", *_shake(24))
    source = write_video_episode(
        shaky,
        tmp_path / "episode.mcap",
        VideoEpisodeSpec(
            duration_s=float(_DURATION_S),
            image_hz=float(_FRAMES_PER_SECOND),
            image_width=320,
            image_height=240,
            camera_name="head_camera",
        ),
    )
    canonical = tmp_path / "episode.canonical.mcap"
    write_canonical_episode(source, canonical, TransformConfig())

    with (
        hflow.Episode(canonical) as episode,
        pytest.raises(
            ValueError,
            match=r"^unstable_min_duration_s must be finite and non-negative$",
        ),
    ):
        camera_stability(episode, unstable_min_duration_s=np.inf)


def test_camera_stability_rejects_bool_min_duration(still_texture: Path, tmp_path: Path) -> None:
    """A boolean minimum unstable duration is rejected instead of being treated as 1."""
    from hflow.testing import VideoEpisodeSpec, write_video_episode
    from hflow.transform import TransformConfig, write_canonical_episode

    shaky = _render_camera_path(still_texture, tmp_path / "shaky.mp4", *_shake(24))
    source = write_video_episode(
        shaky,
        tmp_path / "episode.mcap",
        VideoEpisodeSpec(
            duration_s=float(_DURATION_S),
            image_hz=float(_FRAMES_PER_SECOND),
            image_width=320,
            image_height=240,
            camera_name="head_camera",
        ),
    )
    canonical = tmp_path / "episode.canonical.mcap"
    write_canonical_episode(source, canonical, TransformConfig())

    with (
        hflow.Episode(canonical) as episode,
        pytest.raises(
            ValueError,
            match=r"^unstable_min_duration_s must be finite and non-negative$",
        ),
    ):
        camera_stability(episode, unstable_min_duration_s=True)


def test_the_check_knobs_raise_the_bar_without_changing_the_rate_measurements(
    still_texture: Path, tmp_path: Path
) -> None:
    """A shake threshold above the footage's rates leaves nothing unstable, and
    a minimum duration longer than the episode drops every interval while
    the share it was cut from still reports the raw rule."""
    from hflow.testing import VideoEpisodeSpec, write_video_episode
    from hflow.transform import TransformConfig, write_canonical_episode

    shaky = _render_camera_path(still_texture, tmp_path / "shaky.mp4", *_shake(24))
    source = write_video_episode(
        shaky,
        tmp_path / "episode.mcap",
        VideoEpisodeSpec(
            duration_s=float(_DURATION_S),
            image_hz=float(_FRAMES_PER_SECOND),
            image_width=320,
            image_height=240,
            camera_name="head_camera",
        ),
    )
    canonical = tmp_path / "episode.canonical.mcap"
    write_canonical_episode(source, canonical, TransformConfig())
    with hflow.Episode(canonical) as episode:
        camera_topic = episode.cameras[0]
        baseline = camera_stability(episode)
        above_every_rate = camera_stability(episode, shake_threshold_dps=1e6)
        longer_than_the_episode = camera_stability(
            episode, unstable_min_duration_s=float(_DURATION_S) + 1.0
        )

    assert baseline.intervals
    assert above_every_rate.intervals == []
    assert above_every_rate.measurements[f"{camera_topic}/unstable_share"] == 0.0
    assert longer_than_the_episode.intervals == []
    assert (
        longer_than_the_episode.measurements[f"{camera_topic}/unstable_share"]
        == baseline.measurements[f"{camera_topic}/unstable_share"]
    )
    # The rates describe the footage, not the knobs.
    for key in ("shake_rate_p50_dps", "shake_rate_p95_dps"):
        assert (
            above_every_rate.measurements[f"{camera_topic}/{key}"]
            == (baseline.measurements[f"{camera_topic}/{key}"])
        )


# ``luma_frames`` itself is tested in tests/test_ffmpeg.py, beside the other
# ffmpeg helpers: it needs only numpy, so keeping it here would have skipped it
# whenever the motion extra is absent.
