"""Media and canonical byte-serving: content, Range tolerance, containment."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from hflow_ui import UiSettings, create_app
from ui_test_fixtures import STAMPS, PopulatedWorkspace

import hflow
from hflow.catalog import Catalog, CheckRunRow


def test_media_bytes_are_served_with_content_type(
    api: TestClient, populated_workspace: PopulatedWorkspace
) -> None:
    response = api.get(f"/api/v1/episodes/{populated_workspace.ok_episode_id}/media/wrist_cam")
    assert response.status_code == 200
    assert response.content == populated_workspace.contact_sheet_file.read_bytes()
    assert response.headers["content-type"] == "image/jpeg"
    # An inert image renders inline, but every media response is still hardened.
    assert "attachment" not in response.headers.get("content-disposition", "")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "default-src 'none'" in response.headers["content-security-policy"]


@pytest.fixture()
def mixed_media_workspace(tmp_path: Path) -> tuple[TestClient, str]:
    """A workspace with .jpg, .html, and .svg contact-sheet artifacts, all
    inside the data root."""
    data_root = tmp_path / "data"
    episodes_dir = data_root / "episodes"
    episodes_dir.mkdir(parents=True)
    media_dir = data_root / "media"
    media_dir.mkdir()
    (media_dir / "frame.jpg").write_bytes(b"\xff\xd8\xff\xe0 jpeg \xff\xd9")
    (media_dir / "report.html").write_text("<script>alert(document.cookie)</script>")
    (media_dir / "sheet.svg").write_text(
        "<svg xmlns='http://www.w3.org/2000/svg'><script>alert(1)</script></svg>"
    )
    catalog = Catalog(data_root / "catalog")
    canonical = episodes_dir / "a.canonical.mcap"
    canonical.write_bytes(b"canonical a")
    result = catalog.append_episode(
        canonical_path=canonical,
        stamps=STAMPS,
        episode_metadata={"task": "fold", "operator": "alice", "embodiment": "arm-1"},
        check_rows=[
            CheckRunRow(
                check_name="media/contact_sheet",
                check_version="v1",
                critical=False,
                status=hflow.CheckStatus.MEASURED,
                duration_s=0.01,
                measurements={
                    "artifact/frame": str(media_dir / "frame.jpg"),
                    "artifact/report": str(media_dir / "report.html"),
                    "artifact/sheet": str(media_dir / "sheet.svg"),
                },
            )
        ],
    )
    client = TestClient(create_app(UiSettings(data_root=str(data_root), token=None)))
    return client, result.episode_id


def test_inert_image_is_served_inline(mixed_media_workspace: tuple[TestClient, str]) -> None:
    client, episode_id = mixed_media_workspace
    response = client.get(f"/api/v1/episodes/{episode_id}/media/frame")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert "attachment" not in response.headers.get("content-disposition", "")
    assert response.headers["x-content-type-options"] == "nosniff"


def test_html_artifact_is_forced_to_download_not_rendered(
    mixed_media_workspace: tuple[TestClient, str],
) -> None:
    client, episode_id = mixed_media_workspace
    response = client.get(f"/api/v1/episodes/{episode_id}/media/report")
    assert response.status_code == 200
    # Never text/html on the UI's own origin: opaque bytes, forced download.
    assert response.headers["content-type"] == "application/octet-stream"
    assert "attachment" in response.headers["content-disposition"]
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "default-src 'none'" in response.headers["content-security-policy"]


def test_svg_artifact_is_forced_to_download_not_rendered(
    mixed_media_workspace: tuple[TestClient, str],
) -> None:
    client, episode_id = mixed_media_workspace
    response = client.get(f"/api/v1/episodes/{episode_id}/media/sheet")
    assert response.status_code == 200
    # image/svg+xml is active content; it must not render inline.
    assert response.headers["content-type"] == "application/octet-stream"
    assert "attachment" in response.headers["content-disposition"]
    assert response.headers["x-content-type-options"] == "nosniff"


def test_a_range_request_does_not_500(
    api: TestClient, populated_workspace: PopulatedWorkspace
) -> None:
    response = api.get(
        f"/api/v1/episodes/{populated_workspace.ok_episode_id}/media/wrist_cam",
        headers={"Range": "bytes=0-3"},
    )
    assert response.status_code in (200, 206)
    if response.status_code == 206:
        assert response.content == populated_workspace.contact_sheet_file.read_bytes()[:4]


def test_media_outside_the_data_root_is_403_without_the_path(
    api: TestClient, populated_workspace: PopulatedWorkspace
) -> None:
    response = api.get(f"/api/v1/episodes/{populated_workspace.escaping_episode_id}/media/outside")
    assert response.status_code == 403
    assert str(populated_workspace.outside_media_file) not in response.text
    assert str(populated_workspace.outside_media_file.parent) not in response.text


def test_media_whose_file_vanished_is_404(
    api: TestClient, populated_workspace: PopulatedWorkspace
) -> None:
    response = api.get(f"/api/v1/episodes/{populated_workspace.escaping_episode_id}/media/missing")
    assert response.status_code == 404


def test_unknown_artifact_name_is_404(
    api: TestClient, populated_workspace: PopulatedWorkspace
) -> None:
    response = api.get(
        f"/api/v1/episodes/{populated_workspace.ok_episode_id}/media/no_such_artifact"
    )
    assert response.status_code == 404


def test_media_of_an_unknown_episode_is_404(api: TestClient) -> None:
    assert api.get("/api/v1/episodes/not-an-id/media/wrist_cam").status_code == 404


def test_canonical_bytes_are_served(
    api: TestClient, populated_workspace: PopulatedWorkspace
) -> None:
    response = api.get(f"/api/v1/episodes/{populated_workspace.ok_episode_id}/canonical")
    assert response.status_code == 200
    assert response.content == populated_workspace.canonical_file.read_bytes()


def test_canonical_outside_the_data_root_is_403(
    api: TestClient, populated_workspace: PopulatedWorkspace
) -> None:
    response = api.get(f"/api/v1/episodes/{populated_workspace.escaping_episode_id}/canonical")
    assert response.status_code == 403


def test_canonical_of_an_unknown_episode_is_404(api: TestClient) -> None:
    assert api.get("/api/v1/episodes/not-an-id/canonical").status_code == 404
