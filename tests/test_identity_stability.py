"""A release must not re-version a corpus that processed nothing differently.

Until the identity epoch in :mod:`hflow.behavior`, ``hflow.__version__`` was
folded into ``pipeline_version`` (and thence, via ``provenance/v1``, into the
canonical bytes and so into ``episode_id``), into the MCAP header's library
string (also inside those bytes), and into any step version whose function
referenced a module. A CLI-only patch release therefore invalidated an entire
corpus: ``hflow stale`` listed everything, and re-ingesting byte-identical
sources minted new episode identities instead of deduping.

These tests fail if any of those couplings comes back.
"""

import numpy

import hflow
from hflow.behavior import TRANSFORM_BEHAVIOR_VERSION
from hflow.mcap_writer import _default_library_identifier
from hflow.steps import CheckResult, compute_check_version
from hflow.transform import TransformConfig, compute_pipeline_version

FAKE_RELEASE = "9.9.9"


def _step_version(function: object) -> str:
    return compute_check_version(
        name="probe",
        function=function,  # ty: ignore[invalid-argument-type]
        critical=False,
        requires=frozenset(),
        uses=None,
    )


def _with_faked_release(compute: object) -> tuple[str, str]:
    """Return (value now, value under a faked hflow release)."""
    before = compute()  # ty: ignore[call-non-callable]
    original = hflow.__version__
    hflow.__version__ = FAKE_RELEASE
    try:
        after = compute()  # ty: ignore[call-non-callable]
    finally:
        hflow.__version__ = original
    return before, after


def test_pipeline_version_survives_a_release() -> None:
    before, after = _with_faked_release(lambda: compute_pipeline_version(TransformConfig()))
    assert before == after, "a release must not mark every episode stale"


def test_pipeline_version_still_tracks_transform_configuration() -> None:
    baseline = compute_pipeline_version(TransformConfig())
    assert compute_pipeline_version(TransformConfig(crf=30)) != baseline
    assert compute_pipeline_version(TransformConfig(), {"/derived/speed": "abc"}) != baseline


def test_pipeline_version_tracks_the_transform_behavior_version() -> None:
    """The deliberate lever still works: bumping behavior re-versions."""
    baseline = compute_pipeline_version(TransformConfig())
    import hflow.transform as transform_module

    original = transform_module.TRANSFORM_BEHAVIOR_VERSION
    transform_module.TRANSFORM_BEHAVIOR_VERSION = "2"
    try:
        assert compute_pipeline_version(TransformConfig()) != baseline
    finally:
        transform_module.TRANSFORM_BEHAVIOR_VERSION = original


def test_canonical_bytes_carry_no_release_number() -> None:
    """The MCAP header's library string is inside the hashed bytes."""
    identifier = _default_library_identifier()
    assert hflow.__version__ not in identifier
    assert TRANSFORM_BEHAVIOR_VERSION in identifier

    before, after = _with_faked_release(_default_library_identifier)
    assert before == after, "a release must not change episode_id"


def test_step_referencing_the_hflow_module_survives_a_release() -> None:
    def check_via_module(episode: object) -> object:
        return hflow.CheckResult(measurements={"ok": 1.0})

    before, after = _with_faked_release(lambda: _step_version(check_via_module))
    assert before == after


def test_step_referencing_a_third_party_module_survives_its_upgrade() -> None:
    """The defect was never hflow-specific: numpy upgrades churned too."""

    def check_via_numpy(values: list[float]) -> float:
        return float(numpy.mean(values))

    before = _step_version(check_via_numpy)
    original = numpy.__version__
    numpy.__version__ = "99.0.0"
    try:
        after = _step_version(check_via_numpy)
    finally:
        numpy.__version__ = original
    assert before == after


def test_step_version_still_tracks_what_the_author_wrote() -> None:
    """Author-owned facts must still re-version -- that is the point."""

    def original_threshold(episode: object) -> CheckResult:
        return CheckResult(measurements={"limit": 1.0})

    def changed_threshold(episode: object) -> CheckResult:
        return CheckResult(measurements={"limit": 2.0})

    assert _step_version(original_threshold) != _step_version(changed_threshold)

    captured_limit = 1.0

    def uses_closure(episode: object) -> CheckResult:
        return CheckResult(measurements={"limit": captured_limit})

    with_first_capture = _step_version(uses_closure)
    captured_limit = 2.0
    assert _step_version(uses_closure) != with_first_capture
