import pytest
from pytest import CaptureFixture

from hflow import __version__
from hflow.cli import _build_parser


def test_cli_version(capsys: CaptureFixture) -> None:
    with pytest.raises(SystemExit) as exception:
        _build_parser().parse_args(["--version"])

    assert exception.value.code == 0
    assert f"hflow {__version__}" in capsys.readouterr().out
