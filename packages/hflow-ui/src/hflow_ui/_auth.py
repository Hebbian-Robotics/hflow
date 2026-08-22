"""Session-token auth: the middleware every non-exempt request passes.

The launch token reaches the server one of three ways -- ``?token=`` (the
printed URL), an ``Authorization: Bearer`` header, or the session cookie the
first tokened request sets. Anything else is refused: 401 JSON under
``/api``, a small 401 page elsewhere. ``/api/v1/health`` stays open so a
liveness probe needs no secret. All comparisons are constant-time.
"""

import secrets

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.types import ASGIApp

SESSION_COOKIE_NAME = "hflow_ui_session"

_EXEMPT_PATHS = frozenset({"/api/v1/health"})
_BEARER_PREFIX = "bearer "

_UNAUTHORIZED_PAGE = """<!doctype html>
<html>
  <head><title>HFlow workspace UI &mdash; session token required</title></head>
  <body>
    <h1>401 &mdash; session token required</h1>
    <p>Open the exact URL <code>hflow ui</code> printed when it started (it
    carries a one-time <code>?token=...</code> that logs this browser in), or
    restart <code>hflow ui</code> to mint a new one.</p>
  </body>
</html>
"""


class SessionTokenMiddleware(BaseHTTPMiddleware):
    """Refuses every request that cannot present the launch's session token."""

    def __init__(self, app: ASGIApp, *, session_token: str) -> None:
        super().__init__(app)
        self._session_token = session_token

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        query_token = request.query_params.get("token")
        if query_token is not None and self._token_matches(query_token):
            response = await call_next(request)
            # The printed URL logs the browser in exactly once: from here the
            # cookie carries the session, so in-app links need no token.
            response.set_cookie(
                SESSION_COOKIE_NAME, self._session_token, httponly=True, samesite="lax"
            )
            return response

        presented_token = self._presented_token(request)
        if presented_token is not None and self._token_matches(presented_token):
            return await call_next(request)

        if request.url.path == "/api" or request.url.path.startswith("/api/"):
            return JSONResponse({"detail": "missing or invalid session token"}, status_code=401)
        return HTMLResponse(_UNAUTHORIZED_PAGE, status_code=401)

    def _token_matches(self, presented_token: str) -> bool:
        # Encoded to bytes: compare_digest on str raises for non-ASCII input,
        # and a presented credential is attacker-controlled text.
        return secrets.compare_digest(
            presented_token.encode("utf-8"), self._session_token.encode("utf-8")
        )

    def _presented_token(self, request: Request) -> str | None:
        authorization_header = request.headers.get("authorization")
        if authorization_header is not None and authorization_header.lower().startswith(
            _BEARER_PREFIX
        ):
            return authorization_header[len(_BEARER_PREFIX) :]
        return request.cookies.get(SESSION_COOKIE_NAME)
