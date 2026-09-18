"""Real process boundaries enforce limits without disclosing diagnostics."""

import hashlib
import sys
from pathlib import Path

import pytest

from hflow.ffmpeg import MediaBinarySpec, verify_media_binary
from hflow.ffmpeg._process import MediaToolError, media_input_was_rejected, run_media_command


@pytest.mark.parametrize(
    "program",
    [
        "import time; time.sleep(10)",
        "import sys; sys.stdout.write('x' * 10000)",
        "import sys; sys.stderr.write('secret' * 10000)",
    ],
)
def test_media_process_limits_raise_operational_errors(program: str) -> None:
    with pytest.raises(MediaToolError) as captured:
        run_media_command(
            [sys.executable, "-c", program], timeout_seconds=0.1, maximum_output_bytes=100
        )
    assert "secret" not in str(captured.value)


def test_encoder_failure_is_never_classified_as_unreadable() -> None:
    result = run_media_command(
        [
            sys.executable,
            "-c",
            "import sys; sys.stderr.write(\"Unknown encoder 'private-encoder'\"); sys.exit(1)",
        ],
        timeout_seconds=10,
    )
    with pytest.raises(MediaToolError) as captured:
        media_input_was_rejected(result)
    assert "private-encoder" not in str(captured.value)
    assert "private-encoder" not in repr(result)


def test_offline_verification_executes_the_exact_relative_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "ffmpeg"
    executable.write_text("#!/bin/sh\nprintf 'ffmpeg version reviewed-1\\n'\n")
    executable.chmod(0o755)
    specification = MediaBinarySpec(
        "ffmpeg", hashlib.sha256(executable.read_bytes()).hexdigest(), "reviewed"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", "/nonexistent")
    verified = verify_media_binary(Path("ffmpeg"), specification)
    assert verified.path == executable
    assert verified.version == "reviewed-1"
    executable.write_text("#!/bin/sh\nexit 0\n")
    with pytest.raises(MediaToolError, match="checksum"):
        verify_media_binary(executable, specification)
