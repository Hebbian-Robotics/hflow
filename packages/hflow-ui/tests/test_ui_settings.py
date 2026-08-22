"""UiSettings: the launch values parsed where they are set."""

import pytest
from hflow_ui import UiSettings


def test_a_port_that_cannot_be_served_is_refused_at_the_boundary() -> None:
    """Every unusable port answers the same way, before anything is built.

    Left to ``bind(2)``, an out-of-range port surfaces as an OverflowError
    from inside the free-port probe, and 0 binds happily while the login URL
    the launch prints -- ``http://127.0.0.1:0/`` -- is not dialable by the
    browser it is handed to. Checking the field where it is set is also what
    gives a library caller the same answer as the command line.
    """
    for unusable_port in (-1, 0, 65536, 70000):
        with pytest.raises(ValueError, match="1-65535"):
            UiSettings(data_root="/tmp/does-not-need-to-exist", port=unusable_port)


def test_the_default_launch_is_loopback_and_servable() -> None:
    settings = UiSettings(data_root="/tmp/does-not-need-to-exist")
    assert settings.host == "127.0.0.1"
    assert 1 <= settings.port <= 65535
