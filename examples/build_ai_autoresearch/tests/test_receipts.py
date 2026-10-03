"""A failed filesystem write cannot publish an incomplete experiment receipt."""

import subprocess
import sys
from pathlib import Path

from examples.build_ai_autoresearch.contracts import WorkerStarted, read_record, write_record


def test_failed_write_leaves_no_published_receipt(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    command = (
        "import resource, sys\n"
        "from pathlib import Path\n"
        "from examples.build_ai_autoresearch.contracts import WorkerStarted, write_record\n"
        "resource.setrlimit(resource.RLIMIT_FSIZE, (1, 1))\n"
        "write_record(Path(sys.argv[1]), WorkerStarted(started_monotonic=1.0))\n"
    )
    outcome = subprocess.run(
        [sys.executable, "-c", command, str(receipt)],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert outcome.returncode != 0
    assert not receipt.exists()
    expected = WorkerStarted(started_monotonic=1.0)
    write_record(receipt, expected)
    assert read_record(receipt, WorkerStarted) == expected
