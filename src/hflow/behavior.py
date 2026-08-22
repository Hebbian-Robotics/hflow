"""Processing-behavior versions: what actually changes a corpus's identity.

HFlow's identities are content hashes, and until the identity epoch below
they folded in ``hflow.__version__`` -- the RELEASE number. That made every
release, however unrelated to processing, change three things at once:
``pipeline_version`` (so ``hflow stale`` listed the whole corpus),
``episode_id`` (because the pipeline version is stamped inside the canonical
bytes the id hashes, breaking content-addressed dedupe), and every step
version whose function happened to reference the ``hflow`` module. A CLI fix
or a docs-driven version bump invalidated a petabyte.

The release number is a bad proxy for "does this build process data
differently". This module holds the honest answer instead: a version that a
maintainer bumps DELIBERATELY when canonicalization semantics change -- when
the same input would now produce different canonical bytes.

**Bump ``TRANSFORM_BEHAVIOR_VERSION`` when, and only when, a change makes the
transform write different bytes for the same input**: encoder settings or
defaults, chunking and grouping, timestamp handling, the provenance record's
shape, or a fix to any of them. Bumping it re-versions every corpus exactly
once, which is the cost of telling the truth; NOT bumping it when behavior
changed silently mixes two behaviors under one version, which is worse. When
in doubt, bump.

Deliberately NOT here: an "analysis" behavior version covering what a check
observes. Step versions must not fold in an engine-wide constant, because
``@app.derive`` channel versions flow into ``compute_pipeline_version`` and
therefore into ``episode_id`` -- one engine-wide analysis bump would churn
every derived-channel user's episode identities, reintroducing the defect
this module exists to remove. A change to a built-in check's algorithm is
caught where it belongs: in the step's own content hash, once the step's
source or its captured configuration changes.

The release number is not lost, only demoted from identity to provenance:
:attr:`hflow.PipelineManifest.hflow_version` and the rendered bundle's
``hflow-bundle.json`` both record which build produced a pipeline.
"""

# Canonicalization semantics: bump when the transform would write different
# bytes for the same input. See the module docstring for the rule. Annotated
# as ``str`` rather than inferred as a literal: the whole point is that it
# changes.
TRANSFORM_BEHAVIOR_VERSION: str = "1"

# The identity epoch this build stamps. Epoch 1 folded hflow.__version__ into
# pipeline_version, episode_id, and step versions; epoch 2 folds in only
# author-owned facts plus TRANSFORM_BEHAVIOR_VERSION. Recorded here so a
# reader can explain a one-time re-version rather than guess at it.
IDENTITY_EPOCH: str = "2"
