Closes #379.

The manifest recorded how many episodes were converted but not which ones, so a truncated, missing, or swapped episode was undetectable from the delivery (#379's controlled result: one landing file cut to zero bytes, manifest unchanged). This adds the receipt.

`prepared-manifest.json` moves to schema version 3: every delivered episode carries its published URI, its `content_episode_id` content address, and its byte size. The v2 top-level keys are unchanged, so existing readers keep working, and the entries carry everything a future verify command needs, keeping verify purely additive per the issue direction: the entries are done here, verify is deferred, not rejected.

Per the constraints on the issue: the hash is taken inside `_convert_single_episode` while the canonical file is still on local disk, before `storage.publish`, so a bucket root never downloads its own upload to learn its content id; the recorded URI is the published object (a bucket prefix recipient has no local paths); and `content_episode_id` stays the single hashing implementation, reused rather than forked. The source cache is not hashed; the receipt covers what was delivered, not what it was made from.

The truncation fixture from the issue lands as `test_manifest_content_id_detects_a_truncated_episode`: after truncation, both the size and the content id disagree with the manifest, which is the detection the receipt exists for.

Cost: one linear sha256 read of each canonical file while it is already on local disk, 4.6 ms at fixture scale, about 0.2 s for a 100 MB episode.

Gate: ruff, format, ty clean; 1471 passed / 6 skipped, the single failure (`utc-stats-test-date-collision`) reproduced on the base commit and unrelated, fix already in flight on its own branch.

Refs #379, builds on #377's published URI list.
