"""Session-token auth: the three presentation lanes and every refusal."""

import pytest
from fastapi.testclient import TestClient
from hflow_ui import UiSettings, create_app, new_session_token
from ui_test_fixtures import PopulatedWorkspace

SESSION_TOKEN = "test-session-token-123"


@pytest.fixture()
def tokened_api(populated_workspace: PopulatedWorkspace) -> TestClient:
    settings = UiSettings(data_root=str(populated_workspace.data_root), token=SESSION_TOKEN)
    return TestClient(create_app(settings))


def test_new_session_token_is_long_and_unique() -> None:
    first_token = new_session_token()
    second_token = new_session_token()
    assert first_token != second_token
    assert len(first_token) >= 32


def test_health_needs_no_token(tokened_api: TestClient) -> None:
    response = tokened_api.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_api_without_token_is_401_json(tokened_api: TestClient) -> None:
    response = tokened_api.get("/api/v1/config")
    assert response.status_code == 401
    assert "application/json" in response.headers["content-type"]
    assert "token" in response.json()["detail"]


def test_page_without_token_is_401_html(tokened_api: TestClient) -> None:
    response = tokened_api.get("/")
    assert response.status_code == 401
    assert "text/html" in response.headers["content-type"]


def test_query_token_logs_the_browser_in_with_a_cookie(tokened_api: TestClient) -> None:
    # A valid ?token= is a ONE-TIME login: it 302-redirects to the same path
    # with the token stripped, setting the session cookie -- so the credential
    # never lingers in the URL, history, or the access log afterward.
    response = tokened_api.get(
        "/api/v1/config", params={"token": SESSION_TOKEN}, follow_redirects=False
    )
    assert response.status_code == 302
    assert "token" not in response.headers["location"]
    set_cookie_header = response.headers.get("set-cookie", "")
    assert "hflow_ui_session" in set_cookie_header
    assert "httponly" in set_cookie_header.lower()
    assert "samesite=lax" in set_cookie_header.lower()
    # The TestClient keeps the cookie: the next request needs no token at all.
    followup_response = tokened_api.get("/api/v1/episodes")
    assert followup_response.status_code == 200


def test_query_token_redirect_preserves_other_query_params(tokened_api: TestClient) -> None:
    response = tokened_api.get(
        "/api/v1/episodes",
        params={"token": SESSION_TOKEN, "limit": "10", "order": "asc"},
        follow_redirects=False,
    )
    assert response.status_code == 302
    location = response.headers["location"]
    assert "token" not in location
    assert "limit=10" in location
    assert "order=asc" in location


def test_cookie_is_refused_for_a_cross_site_request(tokened_api: TestClient) -> None:
    # A page on another 127.0.0.1 port is same-SITE, so the browser attaches
    # this cookie; the Origin / Sec-Fetch-Site check refuses it. A Bearer
    # header (which a foreign origin cannot forge) still authenticates.
    tokened_api.cookies.set("hflow_ui_session", SESSION_TOKEN)
    cross_site = tokened_api.get("/api/v1/config", headers={"Sec-Fetch-Site": "cross-site"})
    assert cross_site.status_code == 401
    same_site = tokened_api.get("/api/v1/config", headers={"Sec-Fetch-Site": "same-site"})
    assert same_site.status_code == 401
    foreign_origin = tokened_api.get("/api/v1/config", headers={"Origin": "http://127.0.0.1:3000"})
    assert foreign_origin.status_code == 401
    # The SPA's own same-origin fetch is accepted.
    same_origin = tokened_api.get("/api/v1/config", headers={"Sec-Fetch-Site": "same-origin"})
    assert same_origin.status_code == 200
    # And a Bearer header authenticates regardless of origin.
    bearer_cross_site = tokened_api.get(
        "/api/v1/config",
        headers={"Sec-Fetch-Site": "cross-site", "Authorization": f"Bearer {SESSION_TOKEN}"},
    )
    assert bearer_cross_site.status_code == 200


def test_bearer_header_authenticates(tokened_api: TestClient) -> None:
    response = tokened_api.get(
        "/api/v1/config", headers={"Authorization": f"Bearer {SESSION_TOKEN}"}
    )
    assert response.status_code == 200


def test_wrong_query_token_is_refused(tokened_api: TestClient) -> None:
    assert tokened_api.get("/api/v1/config", params={"token": "wrong"}).status_code == 401


def test_wrong_bearer_token_is_refused(tokened_api: TestClient) -> None:
    response = tokened_api.get("/api/v1/config", headers={"Authorization": "Bearer nope"})
    assert response.status_code == 401


def test_forged_cookie_is_refused(tokened_api: TestClient) -> None:
    tokened_api.cookies.set("hflow_ui_session", "forged-value")
    assert tokened_api.get("/api/v1/config").status_code == 401


def test_disabling_the_token_disables_auth(api: TestClient) -> None:
    assert api.get("/api/v1/config").status_code == 200
