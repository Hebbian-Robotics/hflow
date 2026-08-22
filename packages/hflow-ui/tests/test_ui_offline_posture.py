"""The offline promise: nothing this server serves points at another host.

docs/UI.md's "Trust posture" tells an operator the UI makes no CDN and no
outbound requests -- a claim they act on when deciding to run it on an
air-gapped host or in front of a colleague. FastAPI's built-in Swagger and
ReDoc pages would quietly falsify it (their JS, CSS and favicon come from
cdn.jsdelivr.net and fastapi.tiangolo.com), so ``create_app`` disables both
and serves only ``/api/openapi.json``. These tests are what keeps that true
if someone re-enables a docs page or pastes a font link into a rendered page.
"""

import re

import pytest
from fastapi.testclient import TestClient
from hflow_ui import UiSettings, create_app
from ui_test_fixtures import PopulatedWorkspace

# Every absolute URL, whatever quoting or markup surrounds it.
_ABSOLUTE_URL = re.compile(r"https?://[^\s\"'<>)]+", re.IGNORECASE)

# The surfaces this package renders itself, plus the schema it publishes.
# /api/docs and /api/redoc are here on purpose: they must stay unserved.
_SERVER_OWNED_PATHS = (
    "/",
    "/episodes/some-episode-id",
    "/api/docs",
    "/api/redoc",
    "/api/openapi.json",
    "/api/v1/health",
    "/api/v1/config",
)


def _referenced_hosts(body: str) -> list[str]:
    return _ABSOLUTE_URL.findall(body)


@pytest.mark.parametrize("path", _SERVER_OWNED_PATHS)
def test_no_served_surface_references_an_external_host(api: TestClient, path: str) -> None:
    response = api.get(path)
    assert _referenced_hosts(response.text) == [], f"{path} points at another host"
    assert "cdn.jsdelivr" not in response.text


def test_the_interactive_docs_pages_are_not_served(api: TestClient) -> None:
    # Not "they happen to 404": FastAPI serves these by default, so a config
    # change that re-enables either one must fail here rather than in a
    # customer's browser.
    for docs_path in ("/api/docs", "/api/redoc"):
        assert api.get(docs_path).status_code == 404


def test_the_openapi_schema_is_the_published_contract(api: TestClient) -> None:
    schema = api.get("/api/openapi.json").json()
    assert schema["info"]["title"] == "HFlow workspace UI"
    assert "/api/v1/episodes" in schema["paths"]


def test_the_unauthenticated_page_is_self_contained(
    populated_workspace: PopulatedWorkspace,
) -> None:
    # The 401 HTML page is the one surface an unauthenticated browser reaches,
    # so it is the one most tempting to dress up with a hosted stylesheet.
    tokened_api = TestClient(
        create_app(UiSettings(data_root=str(populated_workspace.data_root), token="offline-token"))
    )
    response = tokened_api.get("/")
    assert response.status_code == 401
    assert _referenced_hosts(response.text) == []
