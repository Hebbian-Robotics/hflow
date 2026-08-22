"""The access log must never carry the session token.

``serve`` prints a one-time login URL (``/?token=...``) and anticipates stdout
being piped to a supervisor's log, so uvicorn's access line -- which logs the
full path, query string included -- would otherwise persist a live 256-bit
credential. ``_QueryStringStrippingAccessFormatter`` drops the query string;
these tests are the only thing standing between that intent and a regression,
since the formatter runs inside uvicorn and never in a TestClient request.
"""

import logging

from hflow_ui.server import _access_log_config, _QueryStringStrippingAccessFormatter
from uvicorn.config import LOGGING_CONFIG

# uvicorn's access record: (client_addr, method, full_path, http_version, status).
_LOGIN_REQUEST_ARGS = ("127.0.0.1:52814", "GET", "/api/v1/config?token=SECRET", "1.1", 200)


def _formatted_access_line(*record_args: object) -> str:
    formatter = _QueryStringStrippingAccessFormatter(
        fmt=LOGGING_CONFIG["formatters"]["access"]["fmt"], use_colors=False
    )
    record = logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='%s - "%s %s HTTP/%s" %d',
        args=record_args,
        exc_info=None,
    )
    return formatter.format(record)


def test_the_login_token_never_reaches_the_access_line() -> None:
    access_line = _formatted_access_line(*_LOGIN_REQUEST_ARGS)
    assert "SECRET" not in access_line
    assert "token" not in access_line
    # The line stays useful: method, path and status all survive.
    assert "GET /api/v1/config" in access_line
    assert "200" in access_line


def test_other_query_parameters_are_dropped_with_it() -> None:
    # Whole-query-string removal, not token-specific redaction: any future
    # credential-bearing parameter is covered without editing this code.
    access_line = _formatted_access_line(
        "127.0.0.1:52814", "GET", "/api/v1/episodes?limit=10&order=asc", "1.1", 200
    )
    assert "limit" not in access_line
    assert "GET /api/v1/episodes" in access_line


def _configured_access_formatter(log_config: dict[str, object]) -> str:
    """The dotted path a logging dictConfig would instantiate for access lines."""
    formatters = log_config["formatters"]
    assert isinstance(formatters, dict)
    return str(formatters["access"]["()"])


def test_the_access_log_config_swaps_the_formatter_without_touching_uvicorns() -> None:
    # A shallow copy would rewrite uvicorn's module-level LOGGING_CONFIG for
    # every other server in the process, so the deep copy is load-bearing.
    assert _configured_access_formatter(_access_log_config()).endswith(
        _QueryStringStrippingAccessFormatter.__name__
    )
    assert _configured_access_formatter(LOGGING_CONFIG) == "uvicorn.logging.AccessFormatter"
