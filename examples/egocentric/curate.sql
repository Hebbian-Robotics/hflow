-- Training-manifest cut: checked-and-healthy episodes only. `status = 'ok'`
-- rather than `!= 'quarantined'`, which also keeps episodes whose critical
-- check crashed and so were never actually checked. Measurement keys recorded by
-- checks (black_frame_pct, freeze_total_s, ...) are plain columns on the wide
-- `episodes` view -- see docs/CATALOG.md for the views and query patterns.
SELECT
    episode_id,
    uri,
    task,
    black_frame_pct,
    freeze_total_s,
    luma_avg_mean
FROM episodes
WHERE status = 'ok'
  AND black_frame_pct < 5.0
  AND freeze_total_s < 2.0
ORDER BY task, episode_id
