"""Camera-shake measurement: it must tell a shaking camera from a moving scene.

Fixtures crop a moving window out of one STILL textured image, so the scene is
byte-identical across cases and only the camera path differs. Using an animated
source instead would make scene motion indistinguishable from camera motion --
which is the whole question the check exists to answer, so a fixture that blurs
it proves nothing.
"""

import subprocess
from pathlib import Path

import pytest

import hflow
from hflow.checks import camera_stability
from hflow.ffmpeg import ffmpeg_path, luma_frames
from hflow.motion import CameraMotion, measure_camera_motion

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


def _measure(video: Path) -> CameraMotion:
    motion = measure_camera_motion(video, frames_per_second=_FRAMES_PER_SECOND)
    assert motion is not None, "the fixture must be measurable"
    return motion


def test_a_static_camera_reports_no_unstable_footage(still_texture: Path, tmp_path: Path) -> None:
    motion = _measure(_render_camera_path(still_texture, tmp_path / "static.mp4", "160", "120"))
    assert motion.unstable_share == 0.0
    # Well under the instrument's own resolution: nothing moved.
    assert motion.shake_rate_p50_deg_per_s < motion.resolution_floor_deg_per_s


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
        rates_per_pixel.append(motion.shake_rate_p50_deg_per_s / amplitude_px)

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
    assert motion.resolution_floor_deg_per_s > 0.0
    assert motion.shake_rate_p50_deg_per_s < motion.resolution_floor_deg_per_s
    assert motion.unstable_share == 0.0


def test_every_pair_is_accounted_for_as_measured_or_unclassified(
    still_texture: Path, tmp_path: Path
) -> None:
    """A share over an unstated denominator is not a measurement."""
    motion = _measure(_render_camera_path(still_texture, tmp_path / "pan.mp4", "160+30*t", "120"))
    pair_count = _FRAMES_PER_SECOND * _DURATION_S - 1
    accounted_s = motion.measured_s + motion.unclassified_s
    assert accounted_s == pytest.approx(pair_count / _FRAMES_PER_SECOND, abs=0.05)
    assert motion.unstable_s <= motion.measured_s


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


def test_luma_frames_streams_every_frame_without_re_encoding(
    still_texture: Path, tmp_path: Path
) -> None:
    video = _render_camera_path(still_texture, tmp_path / "static.mp4", "160", "120")
    with luma_frames(video) as frames:
        shapes = [frame.shape for frame in frames]
    assert len(shapes) == _FRAMES_PER_SECOND * _DURATION_S
    assert set(shapes) == {(240, 320)}


def test_luma_frames_reaps_ffmpeg_when_the_caller_stops_early(
    still_texture: Path, tmp_path: Path
) -> None:
    """Abandoning the iterator must not leave a decode running."""
    video = _render_camera_path(still_texture, tmp_path / "static.mp4", "160", "120")

    # Reading one frame and leaving the block is the abandonment case: ffmpeg
    # then fails writing to a closed pipe, which must not be reported as a
    # decode failure.
    with luma_frames(video) as frames:
        first_frame = next(iter(frames))
    assert first_frame.shape == (240, 320)
