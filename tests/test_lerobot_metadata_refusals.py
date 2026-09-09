"""Regression coverage for LeRobot metadata refusal boundaries (#405)."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.request import Request

import pytest

import hflow.importers.lerobot as prep

_REPO = "fake/repo"
_SHA = "abcdef1234567890"


def _exactly(message: str) -> str:
    """A ``match=`` pattern pinning the whole message, metacharacters and all.

    #405 asks for these to be byte-identical, and several contain a ``.``
    (``meta/info.json``), which unescaped would also match ``meta/infoXjson``.
    """
    return rf"^{re.escape(message)}$"


class _Response:
    def __init__(self, body: bytes, *, link: str | None = None) -> None:
        self._body = body
        self.headers = {} if link is None else {"Link": link}

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def _stub_repo_info(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        prep,
        "_hf_repo_info",
        lambda _repo, _revision: {"sha": _SHA, "license": "apache-2.0"},
    )


def _assert_no_dataset_output(output_dir: Path) -> None:
    assert not (output_dir / "landing").exists()
    assert not (output_dir / "prepared-manifest.json").exists()


def _import(output_dir: Path) -> None:
    prep.import_lerobot_dataset(dataset_repo=_REPO, output_dir=output_dir)


def test_import_refuses_repository_without_lerobot_v3_info(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_repo_info(monkeypatch)
    monkeypatch.setattr(prep, "_hf_tree", lambda _repo, _revision, _path: [])
    output_dir = tmp_path / "out"

    with pytest.raises(
        RuntimeError, match=_exactly("meta/info.json not found; not a LeRobot v3 repository")
    ):
        _import(output_dir)

    _assert_no_dataset_output(output_dir)


def test_import_refuses_info_json_that_is_not_an_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_repo_info(monkeypatch)
    monkeypatch.setattr(
        prep,
        "_hf_tree",
        lambda _repo, _revision, _path: [{"path": "meta/info.json", "type": "file"}],
    )
    monkeypatch.setattr(prep.urllib.request, "urlopen", lambda *_args, **_kwargs: _Response(b"[]"))
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match=_exactly("LeRobot meta/info.json is not a JSON object")):
        _import(output_dir)

    _assert_no_dataset_output(output_dir)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("data_path", "", "LeRobot meta/info.json must define a non-empty data_path template"),
        ("video_path", "", "LeRobot meta/info.json must define a non-empty video_path template"),
    ],
)
def test_import_refuses_empty_path_templates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    message: str,
) -> None:
    _stub_repo_info(monkeypatch)
    info = {
        "fps": 30,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {},
    }
    info[field] = value
    monkeypatch.setattr(prep, "_fetch_info_json", lambda _repo, _revision, _cache: info)
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match=_exactly(message)):
        _import(output_dir)

    _assert_no_dataset_output(output_dir)


@pytest.mark.parametrize(
    ("field", "template", "match_pattern"),
    [
        # Unknown field: the plausible accident, a template indexing episodes.
        (
            "data_path",
            "data/{episode_index:03d}/f.parquet",
            _exactly(
                "LeRobot meta/info.json has an invalid data_path template "
                "'data/{episode_index:03d}/f.parquet': unknown field 'episode_index'"
            ),
        ),
        # A video-template field is unknown to data_path: proves each template
        # is checked against its own field set, not a shared union.
        (
            "data_path",
            "data/{video_key}/f.parquet",
            _exactly(
                "LeRobot meta/info.json has an invalid data_path template "
                "'data/{video_key}/f.parquet': unknown field 'video_key'"
            ),
        ),
        (
            "data_path",
            "data/{chunk_index:03d/f.parquet",
            _exactly(
                "LeRobot meta/info.json has an invalid data_path template "
                "'data/{chunk_index:03d/f.parquet': unmatched '{' in format spec"
            ),
        ),
        # The format-spec detail differs across Python versions (3.11 omits
        # the specifier and type), so this one pins the stable prefix only.
        (
            "data_path",
            "data/{chunk_index:qq}/f.parquet",
            r"^"
            + re.escape(
                "LeRobot meta/info.json has an invalid data_path template "
                "'data/{chunk_index:qq}/f.parquet': Invalid format specifier"
            ),
        ),
        (
            "data_path",
            "data/{0}/f.parquet",
            _exactly(
                "LeRobot meta/info.json has an invalid data_path template "
                "'data/{0}/f.parquet': positional fields are not supported"
            ),
        ),
        (
            "data_path",
            "data/}chunk/f.parquet",
            _exactly(
                "LeRobot meta/info.json has an invalid data_path template "
                "'data/}chunk/f.parquet': Single '}' encountered in format string"
            ),
        ),
        (
            "video_path",
            "videos/{episode_index}/f.mp4",
            _exactly(
                "LeRobot meta/info.json has an invalid video_path template "
                "'videos/{episode_index}/f.mp4': unknown field 'episode_index'"
            ),
        ),
        (
            "video_path",
            "videos/{0}/f.mp4",
            _exactly(
                "LeRobot meta/info.json has an invalid video_path template "
                "'videos/{0}/f.mp4': positional fields are not supported"
            ),
        ),
    ],
)
def test_import_refuses_invalid_path_templates_before_listing_episodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    template: str,
    match_pattern: str,
) -> None:
    """A template ``str.format`` would reject is refused at the parse boundary.

    #471: these previously passed the boundary and raised a bare
    ``KeyError``/``ValueError``/``IndexError`` out of episode conversion,
    after every episode metadata shard had been downloaded. The second
    assertion pins the "before" part: an invalid template must not cost a
    listing of ``meta/episodes``.
    """
    _stub_repo_info(monkeypatch)
    info = {
        "fps": 30,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {},
    }
    info[field] = template
    monkeypatch.setattr(prep, "_fetch_info_json", lambda _repo, _revision, _cache: info)
    listed_paths: list[str] = []

    def recording_hf_tree(_repo: str, _revision: str, path: str) -> list[dict[str, str]]:
        listed_paths.append(path)
        return []

    monkeypatch.setattr(prep, "_hf_tree", recording_hf_tree)
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match=match_pattern):
        _import(output_dir)

    assert listed_paths == []
    _assert_no_dataset_output(output_dir)


def test_import_accepts_default_video_path_template_when_field_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Omitting ``video_path`` falls back to the default, which passes the boundary.

    The boundary validates the template that conversion will actually format,
    default included, so the import proceeds past ``info.json`` parsing and
    fails only at the (empty) episode listing.
    """
    _stub_repo_info(monkeypatch)
    monkeypatch.setattr(
        prep,
        "_fetch_info_json",
        lambda _repo, _revision, _cache: {
            "fps": 30,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "features": {},
        },
    )
    monkeypatch.setattr(prep, "_hf_tree", lambda _repo, _revision, _path: [])
    output_dir = tmp_path / "out"

    with pytest.raises(RuntimeError, match=_exactly("no meta/episodes parquet files found")):
        _import(output_dir)

    _assert_no_dataset_output(output_dir)


def test_import_refuses_info_json_without_features_before_listing_episodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal fires at the parse boundary, not after a shard download.

    #433 moved this ahead of episode discovery. Asserting only the message
    would pass either way, so the point of the test is the second assertion:
    a corpus with no ``features`` must not cost a listing of ``meta/episodes``.
    """
    _stub_repo_info(monkeypatch)
    monkeypatch.setattr(
        prep,
        "_fetch_info_json",
        lambda _repo, _revision, _cache: {
            "fps": 30,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        },
    )
    listed_paths: list[str] = []

    def recording_hf_tree(_repo: str, _revision: str, path: str) -> list[dict[str, str]]:
        listed_paths.append(path)
        return []

    monkeypatch.setattr(prep, "_hf_tree", recording_hf_tree)
    output_dir = tmp_path / "out"

    with pytest.raises(
        ValueError, match=_exactly("LeRobot meta/info.json must define a features object")
    ):
        _import(output_dir)

    assert listed_paths == []
    _assert_no_dataset_output(output_dir)


def test_import_refuses_repository_without_episode_parquets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_repo_info(monkeypatch)
    monkeypatch.setattr(
        prep,
        "_fetch_info_json",
        lambda _repo, _revision, _cache: {
            "fps": 30,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": {},
        },
    )
    monkeypatch.setattr(prep, "_hf_tree", lambda _repo, _revision, _path: [])
    output_dir = tmp_path / "out"

    with pytest.raises(RuntimeError, match=_exactly("no meta/episodes parquet files found")):
        _import(output_dir)

    _assert_no_dataset_output(output_dir)


def test_import_refuses_tree_response_that_is_not_a_list_of_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_repo_info(monkeypatch)
    monkeypatch.setattr(prep.urllib.request, "urlopen", lambda *_args, **_kwargs: _Response(b"{}"))
    output_dir = tmp_path / "out"

    with pytest.raises(
        ValueError,
        match=_exactly("Hugging Face tree response for 'meta' is not a list of objects"),
    ):
        _import(output_dir)

    _assert_no_dataset_output(output_dir)


def test_import_refuses_repeated_pagination_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_repo_info(monkeypatch)
    initial_url = f"https://huggingface.co/api/datasets/{_REPO}/tree/{_SHA}/meta?recursive=true"

    fetched_urls: list[str] = []

    def fake_urlopen(request: Request, **_kwargs: object) -> _Response:
        assert request.full_url == initial_url
        fetched_urls.append(request.full_url)
        # Bounded on purpose. Without the visited-URL guard this server would
        # feed the loop its own URL forever, and the test would hang rather
        # than fail: CI would report nothing and a human would wait. Dropping
        # the `next` link after a few passes lets a guard-less loop terminate
        # and fail on the missing refusal instead.
        if len(fetched_urls) > 4:
            return _Response(b"[]")
        return _Response(b"[]", link=f'<{initial_url}>; rel="next"')

    monkeypatch.setattr(prep.urllib.request, "urlopen", fake_urlopen)
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="repeated an already fetched pagination URL"):
        _import(output_dir)

    # The guard fires on the second pass, before a second fetch: one request
    # went out, not five. Without this the bound above could be doing the
    # stopping and the test would still pass.
    assert fetched_urls == [initial_url]
    _assert_no_dataset_output(output_dir)


def test_import_refuses_invalid_pagination_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_repo_info(monkeypatch)
    invalid_url = "https://[invalid"

    def fake_urlopen(_request: Request, **_kwargs: object) -> _Response:
        return _Response(b"[]", link=f'<{invalid_url}>; rel="next"')

    monkeypatch.setattr(prep.urllib.request, "urlopen", fake_urlopen)
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="contains an invalid pagination URL"):
        _import(output_dir)

    _assert_no_dataset_output(output_dir)
