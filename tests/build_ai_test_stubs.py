"""Shared fakes for the hosted Build AI check transport and its retry loop."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from functools import partial
from types import TracebackType

import httpx2
import pytest
from tenacity import AsyncRetrying

import hflow

HOSTED_CHECK_LOGGER_NAME = "hflow.build_ai_vlm_checks"


class StubHostedResponse:
    """A streamed hosted-service response whose body is ``payload`` as JSON."""

    def __init__(self, payload: object) -> None:
        self.headers: dict[str, str] = {}
        self._body = json.dumps(payload).encode("utf-8")

    async def __aenter__(self) -> StubHostedResponse:
        return self

    async def __aexit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    async def aiter_raw(self) -> AsyncIterator[bytes]:
        yield self._body


def stub_hosted_stream(
    monkeypatch: pytest.MonkeyPatch, next_response: Callable[[], httpx2.Response]
) -> None:
    """Answer each hosted request with ``next_response()``, closed after use.

    ``next_response`` may raise instead, to stand in for a transport failure.
    """

    @asynccontextmanager
    async def respond(
        *_arguments: object, **_keyword_arguments: object
    ) -> AsyncIterator[httpx2.Response]:
        response = next_response()
        try:
            yield response
        finally:
            await response.aclose()

    monkeypatch.setattr(httpx2.AsyncClient, "stream", staticmethod(respond))


def hosted_retry_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """The "retry scheduled" records the hosted check logged."""
    return [
        record
        for record in caplog.records
        if record.name == HOSTED_CHECK_LOGGER_NAME
        and "hosted check retry scheduled" in record.getMessage()
    ]


def record_retry_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace the retry loop's sleep with one that records each delay and returns."""
    sleeps: list[float] = []

    async def record_delay(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(
        hflow.build_ai_vlm_checks, "AsyncRetrying", partial(AsyncRetrying, sleep=record_delay)
    )
    return sleeps
