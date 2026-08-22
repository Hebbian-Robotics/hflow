"""Session-token auth: the middleware every non-exempt request passes.

The launch token reaches the server three ways -- ``?token=`` (the printed
login URL), an ``Authorization: Bearer`` header (the SPA's XHR, from its
in-memory session store), or the session cookie the login exchange sets.

Two hardening rules hold here:

- ``?token=`` is a ONE-TIME login credential: a valid query token sets the
  cookie and 302-redirects to the same path with the parameter stripped, so
  the secret never lingers in the address bar, browser history, ``Referer``,
  or the server's access log across the rest of the session.
- The cookie is honored only for SAME-ORIGIN requests. Cookies are not
  port-scoped, so a page on another 127.0.0.1 port is same *site* and its
  requests carry this cookie; an Origin / ``Sec-Fetch-Site`` check refuses
  the cookie for anything that is not the exact scheme+host+port the server
  is bound to. A ``Bearer`` header (which a foreign origin cannot forge, not
  holding the token) is always accepted.

``/api/v1/health`` stays open so a liveness probe needs no secret. All token
comparisons are constant-time.
"""

import secrets
from urllib.parse import urlencode

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.types import ASGIApp

SESSION_COOKIE_NAME = "hflow_ui_session"

_EXEMPT_PATHS = frozenset({"/api/v1/health"})
_BEARER_PREFIX = "bearer "
# Sec-Fetch-Site values that are NOT a foreign page driving the browser:
# "same-origin" is the SPA's own fetches; "none" is a direct navigation
# (typed URL, bookmark). "same-site" is deliberately EXCLUDED -- a different
# port on 127.0.0.1 is same-site, which is exactly the cross-port threat.
_SAME_ORIGIN_FETCH_VALUES = frozenset({"same-origin", "none"})

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


def _cookie_credential_trusted(request: Request) -> bool:
    """Whether the session cookie may authenticate THIS request.

    Trusted only when the request is same-origin: the SPA's own fetches
    (``Sec-Fetch-Site: same-origin``) or a direct navigation
    (``Sec-Fetch-Site: none``). A modern browser sends ``Sec-Fetch-Site`` on
    every request, so its absence means a non-browser (or legacy) client,
    which is not the foreign-page threat; we then fall back to an ``Origin``
    match. A cross-port page on the same host is ``same-site`` (refused) and
    its ``Origin`` differs by port (also refused).
    """
    sec_fetch_site = request.headers.get("sec-fetch-site")
    if sec_fetch_site is not None:
        return sec_fetch_site in _SAME_ORIGIN_FETCH_VALUES
    origin = request.headers.get("origin")
    if origin is None:
        return True
    host = request.headers.get("host", "")
    return origin == f"{request.url.scheme}://{host}"


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
            if request.method in ("GET", "HEAD"):
                # Log the browser in exactly once, then strip the token from
                # the URL: the cookie carries the session from here, so the
                # secret never reaches history, Referer, or the access log.
                response: Response = RedirectResponse(
                    url=self._token_stripped_target(request), status_code=302
                )
            else:
                # A non-navigation request that still carries the query token
                # (legacy callers): honor it without a redirect (a 302 would
                # rewrite the method), and set the cookie for continuity.
                response = await call_next(request)
            response.set_cookie(
                SESSION_COOKIE_NAME, self._session_token, httponly=True, samesite="lax"
            )
            return response

        bearer_token = self._presented_bearer_token(request)
        if bearer_token is not None and self._token_matches(bearer_token):
            return await call_next(request)

        cookie_token = request.cookies.get(SESSION_COOKIE_NAME)
        if (
            cookie_token is not None
            and self._token_matches(cookie_token)
            and _cookie_credential_trusted(request)
        ):
            return await call_next(request)

        if request.url.path == "/api" or request.url.path.startswith("/api/"):
            return JSONResponse({"detail": "missing or invalid session token"}, status_code=401)
        return HTMLResponse(_UNAUTHORIZED_PAGE, status_code=401)

    def _token_stripped_target(self, request: Request) -> str:
        remaining = [
            (key, value) for key, value in request.query_params.multi_items() if key != "token"
        ]
        query_string = urlencode(remaining)
        return request.url.path + (f"?{query_string}" if query_string else "")

    def _token_matches(self, presented_token: str) -> bool:
        # Encoded to bytes: compare_digest on str raises for non-ASCII input,
        # and a presented credential is attacker-controlled text.
        return secrets.compare_digest(
            presented_token.encode("utf-8"), self._session_token.encode("utf-8")
        )

    def _presented_bearer_token(self, request: Request) -> str | None:
        authorization_header = request.headers.get("authorization")
        if authorization_header is not None and authorization_header.lower().startswith(
            _BEARER_PREFIX
        ):
            return authorization_header[len(_BEARER_PREFIX) :]
        return None
