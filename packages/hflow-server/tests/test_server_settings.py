"""ServerSettings: the launch values parsed where they are set."""

from enum import IntEnum

import pytest
from hflow_server import ServerSettings


def test_a_port_that_cannot_be_served_is_refused_at_the_boundary() -> None:
    """Every unusable port answers the same way, before anything is built.

    Left to ``bind(2)``, an out-of-range port surfaces as an OverflowError
    from inside the free-port probe, and 0 binds happily while the URL the
    launch prints -- ``http://127.0.0.1:0/`` -- is not dialable by the
    browser it is handed to. Checking the field where it is set is also what
    gives a library caller the same answer as the command line.
    """
    for unusable_port in (-1, 0, 65536, 70000):
        with pytest.raises(ValueError, match="1-65535"):
            ServerSettings(data_root="/tmp/does-not-need-to-exist", port=unusable_port)


def test_the_default_launch_is_loopback_and_servable() -> None:
    settings = ServerSettings(data_root="/tmp/does-not-need-to-exist")
    assert settings.host == "127.0.0.1"
    assert 1 <= settings.port <= 65535


def test_a_port_that_is_not_an_int_is_refused_with_runtimeconfig_s_message() -> None:
    """The type check `hflow.runtime` grew for the same field, in this package too.

    `bool` subclasses int, so ``True`` satisfies ``1 <= port <= 65535`` and
    reaches the URL the launch prints as ``http://127.0.0.1:True``. A ``str``
    is worse than useless: it raises TypeError out of the range comparison,
    which the CLI's ``except ValueError`` does not catch, so a bad ``--port``
    escapes as a traceback. Both are why the type test is ordered first.
    """
    wrong_type_ports: tuple[object, ...] = (True, 8080.0, "4356", None)
    for wrong_type_port in wrong_type_ports:
        with pytest.raises(ValueError, match="port must be an int"):
            # The wrong type is the point: this guards the callers that are not
            # type checked, which is every command line there has ever been.
            ServerSettings(data_root="/tmp/does-not-need-to-exist", port=wrong_type_port)  # ty: ignore


def test_an_intenum_port_is_still_a_port() -> None:
    """`isinstance` and not `type(...) is int`, matching RuntimeConfig.api_port.

    An IntEnum member is an int by every test Python has, and CONTRIBUTING asks
    for typed variants over bare literals, so refusing one would punish the
    style the repo asks for.
    """

    class WorkspacePort(IntEnum):
        DEFAULT = 4356

    settings = ServerSettings(data_root="/tmp/does-not-need-to-exist", port=WorkspacePort.DEFAULT)
    assert settings.port == 4356
