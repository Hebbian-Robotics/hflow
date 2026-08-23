"""Workspace layout/identity, environment-resolved data roots, and the
pipeline manifest -- the seams a hosted deployment drives without editing
pipeline code (docs/HOSTING.md)."""

import json
from pathlib import Path

import pytest

import hflow
from hflow.manifest import PIPELINE_MANIFEST_VERSION
from hflow.storage import LocalStorageRoot
from hflow.workspace import Workspace


class TestWorkspaceLayout:
    def test_workspace_owns_the_data_root_layout(self, tmp_path: Path) -> None:
        workspace = Workspace.parse(tmp_path)
        assert workspace.episodes_root == LocalStorageRoot(tmp_path / "episodes")
        assert workspace.catalog_root == LocalStorageRoot(tmp_path / "catalog")
        assert workspace.test_runs_root == LocalStorageRoot(tmp_path / "test-runs")

    def test_app_routes_its_layout_through_the_workspace(self, tmp_path: Path) -> None:
        app = hflow.App("pipeline", data_root=tmp_path)
        assert app.workspace.catalog_root == LocalStorageRoot(tmp_path / "catalog")


class TestWorkspaceIdentity:
    def test_identity_is_minted_once_and_read_back(self, tmp_path: Path) -> None:
        workspace = Workspace.parse(tmp_path)
        assert workspace.identity() is None

        first_identity = workspace.ensure_identity()
        assert first_identity.workspace_id
        assert (tmp_path / "workspace.json").is_file()

        # A second initializer (a re-run, or another process) gets the SAME
        # identity: the id must survive re-initialization, or it is not one.
        second_identity = Workspace.parse(tmp_path).ensure_identity()
        assert second_identity == first_identity
        assert Workspace.parse(tmp_path).identity() == first_identity

    def test_corrupt_identity_marker_is_refused_loudly(self, tmp_path: Path) -> None:
        (tmp_path / "workspace.json").write_text("not json at all")
        with pytest.raises(ValueError, match="invalid workspace identity"):
            Workspace.parse(tmp_path).identity()


class TestEnvironmentResolvedDataRoot:
    def test_default_stays_local_data_directory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HFLOW_DATA_ROOT", raising=False)
        assert hflow.App("pipeline").data_root == Path("./data")

    def test_environment_variable_supplies_the_root(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("HFLOW_DATA_ROOT", str(tmp_path / "workspace-root"))
        assert hflow.App("pipeline").data_root == tmp_path / "workspace-root"

    def test_environment_variable_accepts_bucket_urls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HFLOW_DATA_ROOT", "gs://tenant-bucket/workspaces/a")
        assert hflow.App("pipeline").data_root == "gs://tenant-bucket/workspaces/a"

    def test_explicit_argument_wins_over_the_environment(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("HFLOW_DATA_ROOT", str(tmp_path / "from-environment"))
        app = hflow.App("pipeline", data_root=tmp_path / "explicit")
        assert app.data_root == tmp_path / "explicit"


class TestPipelineManifest:
    def test_manifest_describes_the_registrations(self, tmp_path: Path) -> None:
        app = hflow.App("kitchen", data_root=tmp_path, endpoints={"judge": "http://judge:8000"})

        @app.check(critical=True)
        def camera_blackout(ep: hflow.Episode) -> hflow.CheckResult:
            return hflow.CheckResult()

        @app.enrich(uses="judge", requires=("gpu",))
        def caption(ep: hflow.Episode) -> hflow.EnrichmentResult:
            return hflow.EnrichmentResult()

        @app.derive("/derived/speed")
        def speed(ep: hflow.Episode) -> hflow.DerivedSeries:
            raise NotImplementedError  # registration only; never executed here

        # Through the JSON round trip: the manifest's serialized form is the
        # contract a service consumes.
        manifest_payload = json.loads(app.manifest().to_json())
        assert manifest_payload["manifest_version"] == PIPELINE_MANIFEST_VERSION
        assert manifest_payload["pipeline_name"] == "kitchen"
        assert manifest_payload["hflow_version"] == hflow.__version__
        assert manifest_payload["pipeline_version"] == app.pipeline_version

        (check_entry,) = manifest_payload["checks"]
        assert check_entry["name"] == "camera_blackout"
        assert check_entry["kind"] == "check"
        assert check_entry["critical"] is True
        assert check_entry["version"] == app.checks[0].version

        (enrichment_entry,) = manifest_payload["enrichments"]
        assert enrichment_entry["name"] == "caption"
        assert enrichment_entry["kind"] == "enrichment"
        assert enrichment_entry["uses"] == "judge"
        assert enrichment_entry["requires"] == ["gpu"]

        (derived_entry,) = manifest_payload["derived_channels"]
        assert derived_entry["topic"] == "/derived/speed"

        assert manifest_payload["endpoint_aliases"] == ["judge"]
        assert manifest_payload["has_transform_override"] is False

    def test_manifest_carries_the_gate_policy_a_step_rejects_on(self, tmp_path: Path) -> None:
        """``critical`` says a gate exists; the gate says what it is. Without
        this a service can only report that some critical check failed, not
        which threshold on which measurement rejected the episode.
        """
        app = hflow.App("kitchen", data_root=tmp_path)

        @app.check(critical=True, gate=hflow.checks.RECOMMENDED_CAMERA_INTEGRITY)
        def camera_health(ep: hflow.Episode) -> hflow.CheckResult:
            return hflow.CheckResult()

        @app.check()
        def ungated(ep: hflow.Episode) -> hflow.CheckResult:
            return hflow.CheckResult()

        payload = json.loads(app.manifest().to_json())
        gated_entry, ungated_entry = payload["checks"]
        assert ungated_entry["gate"] is None
        assert gated_entry["gate"] == {
            "accept_when": [
                {
                    "key_pattern": "*black_frame_pct",
                    "comparison": "at_most",
                    "value": 50.0,
                    "across": "every_key",
                },
                {
                    "key_pattern": "*freeze_total_s",
                    "comparison": "at_most",
                    "value": 2.0,
                    "across": "every_key",
                },
            ]
        }

    def test_manifest_json_round_trips(self, tmp_path: Path) -> None:
        app = hflow.App("kitchen", data_root=tmp_path)
        parsed = json.loads(app.manifest().to_json())
        assert parsed["pipeline_name"] == "kitchen"
        assert parsed["checks"] == []
