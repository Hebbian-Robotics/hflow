"""The fixed runner stops real training processes that exceed their allowance."""

import sys
from pathlib import Path

import pytest
from examples.build_ai_autoresearch.contracts import TrialBudget
from examples.build_ai_autoresearch.training_runner import run_worker


@pytest.mark.parametrize("overrun", [False, True])
def test_worker_completion_and_deadline(overrun: bool, tmp_path: Path) -> None:
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import json, time\n"
        "from pathlib import Path\n"
        f"output = Path({str(tmp_path)!r})\n"
        "(output / 'started.tmp').write_text(json.dumps({'started_monotonic':time.monotonic()}))\n"
        "(output / 'started.tmp').rename(output / 'started.json')\n"
        f"time.sleep({60 if overrun else 0.1})\n"
        "(output / 'completed').write_text('done')\n"
    )
    budget = TrialBudget(training_seconds=1.0, reference_reason="real process budget fixture")
    if overrun:
        with pytest.raises(TimeoutError, match="compute budget"):
            run_worker([sys.executable, str(worker)], tmp_path, budget)
        assert not (tmp_path / "completed").exists()
    else:
        run_worker([sys.executable, str(worker)], tmp_path, budget)
        assert (tmp_path / "completed").read_text() == "done"
