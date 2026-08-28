"""Contact sheet: N timestamped frames composited into one JPEG grid.

Makes episode-level VLM questions work even on single-image models and cuts
vision tokens (one image instead of N). One of the two VLM-adjacent helpers
we ship -- there is deliberately no bundled VLM client.

Implementation: a single ffmpeg invocation over the frame files
(``concat`` + ``scale`` + ``tile``). Timestamp burn-in uses ``drawtext`` when
both the filter and a usable font are available (``fc-match`` or common font
paths); otherwise the sheet is still produced without burn-in and
``ContactSheet.timestamps_burned`` is False -- callers can pass timestamps in
the prompt instead. If more frames are given than ``max_tiles``, frames are
sampled evenly and the drop is reported on the result, never silent.
"""

import math
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

from hflow.ffmpeg._binary import ffmpeg_path

if TYPE_CHECKING:
    from hflow.episode import ExtractedFrame

_COMMON_FONT_PATHS = (
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
    Path("/System/Library/Fonts/Helvetica.ttc"),
)


@dataclass(frozen=True)
class ContactSheet:
    path: Path
    columns: int
    rows: int
    tile_log_times_ns: list[int]
    timestamps_burned: bool
    frames_sampled_from: int


def _find_usable_font_file() -> Path | None:
    fc_match_binary = shutil.which("fc-match")
    if fc_match_binary is not None:
        completed = subprocess.run(
            [fc_match_binary, "--format", "%{file}", "sans"],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            candidate = Path(completed.stdout.strip())
            if candidate.is_file():
                return candidate
    for common_path in _COMMON_FONT_PATHS:
        if common_path.is_file():
            return common_path
    return None


@cache
def _ffmpeg_supports_drawtext(ffmpeg_binary: Path) -> bool:
    completed = subprocess.run(
        [str(ffmpeg_binary), "-hide_banner", "-filters"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return False
    return any(
        len(columns := line.split()) > 1 and columns[1] == "drawtext"
        for line in completed.stdout.splitlines()
    )


def _evenly_sampled_indices(total_count: int, max_count: int) -> list[int]:
    """Up to ``max_count`` indices evenly spread over ``range(total_count)``,
    always including the first and last frame."""
    if total_count <= max_count:
        return list(range(total_count))
    if max_count == 1:
        return [0]
    # The step is >= 1, so rounded values are strictly increasing (no dupes).
    step = (total_count - 1) / (max_count - 1)
    return [round(sample_index * step) for sample_index in range(max_count)]


def _quote_filter_value(value: str) -> str:
    """Quote a value for use inside an ffmpeg filtergraph option.

    ``drawtext`` parses option values after the filtergraph parser, so an
    embedded apostrophe needs one escaping layer for each parser.
    """
    return "'" + value.replace("'", r"'\\\''") + "'"


def _drawtext_filters(
    selected_frames: "Sequence[ExtractedFrame]", font_file: Path, tile_width: int
) -> list[str]:
    """One ``drawtext`` per tile, each enabled only on its own frame index."""
    font_size = max(12, tile_width // 14)
    first_log_time_ns = selected_frames[0].log_time_ns
    filters: list[str] = []
    for frame_index, frame in enumerate(selected_frames):
        relative_seconds = (frame.log_time_ns - first_log_time_ns) / 1e9
        filters.append(
            "drawtext="
            f"fontfile={_quote_filter_value(str(font_file))}:"
            f"text={_quote_filter_value(f'+{relative_seconds:.1f}s')}:"
            f"x=6:y=6:fontsize={font_size}:fontcolor=white:"
            "borderw=2:bordercolor=black:"
            f"enable={_quote_filter_value(f'eq(n,{frame_index})')}"
        )
    return filters


def _write_concat_list(selected_frames: "Sequence[ExtractedFrame]", list_path: Path) -> None:
    # The concat demuxer's quoting: single-quoted, embedded quotes closed-escaped-reopened.
    lines = [
        "file '" + str(frame.path.resolve()).replace("'", "'\\''") + "'"
        for frame in selected_frames
    ]
    list_path.write_text("\n".join(lines) + "\n")


def contact_sheet(
    frames: "Sequence[ExtractedFrame]",
    output: Path,
    *,
    columns: int = 4,
    tile_width: int = 320,
    max_tiles: int = 24,
) -> ContactSheet:
    """Composite ``frames`` (from ``Episode.frames()``) into one JPEG grid."""
    if not frames:
        raise ValueError("contact_sheet needs at least one frame")
    if not isinstance(columns, int) or isinstance(columns, bool):
        raise ValueError(f"columns must be an int, got {columns!r}")
    if columns < 1:
        raise ValueError(f"columns must be >= 1, got {columns}")
    if not isinstance(tile_width, int) or isinstance(tile_width, bool):
        raise ValueError(f"tile_width must be an int, got {tile_width!r}")
    if tile_width < 1:
        raise ValueError(f"tile_width must be >= 1, got {tile_width}")
    if not isinstance(max_tiles, int) or isinstance(max_tiles, bool):
        raise ValueError(f"max_tiles must be an int, got {max_tiles!r}")
    if max_tiles < 1:
        raise ValueError(f"max_tiles must be >= 1, got {max_tiles}")
    selected_frames = [frames[index] for index in _evenly_sampled_indices(len(frames), max_tiles)]
    rows = math.ceil(len(selected_frames) / columns)

    ffmpeg_binary = ffmpeg_path()
    font_file = _find_usable_font_file()
    timestamps_burned = font_file is not None and _ffmpeg_supports_drawtext(ffmpeg_binary)
    filter_chain: list[str] = [f"scale={tile_width}:-1"]
    if timestamps_burned:
        assert font_file is not None
        filter_chain.extend(_drawtext_filters(selected_frames, font_file, tile_width))
    filter_chain.append(f"tile={columns}x{rows}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="contact-sheet-") as staging_dir_name:
        concat_list_path = Path(staging_dir_name) / "frames.txt"
        _write_concat_list(selected_frames, concat_list_path)
        command = [
            str(ffmpeg_binary),
            "-hide_banner",
            "-nostats",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list_path),
            "-vf",
            ",".join(filter_chain),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        stderr_tail = "\n".join(completed.stderr.strip().splitlines()[-5:])
        raise RuntimeError(f"ffmpeg contact sheet failed for {output}: {stderr_tail}")

    return ContactSheet(
        path=output,
        columns=columns,
        rows=rows,
        tile_log_times_ns=[frame.log_time_ns for frame in selected_frames],
        timestamps_burned=timestamps_burned,
        frames_sampled_from=len(frames),
    )
