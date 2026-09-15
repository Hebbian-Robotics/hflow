SELECT
  episode_id,
  metadata_json,
  (SELECT COUNT(*) FROM episodes) AS total_episodes,
  (SELECT COUNT(*) FROM episodes WHERE status = 'ok') AS ok_episodes,
  (SELECT COUNT(*) FROM episodes WHERE "/observation.images.up/decoded_frame_count" IS NOT NULL AND "/observation.images.side/decoded_frame_count" IS NOT NULL) AS fully_measured_episodes
FROM episodes
WHERE status = 'ok' AND "/observation.images.up/decoded_frame_count" = "/observation.images.up/message_count" AND "/observation.images.side/decoded_frame_count" = "/observation.images.side/message_count" AND coalesce("/observation.images.up/freeze_total_s", 0) = 0 AND coalesce("/observation.images.side/freeze_total_s", 0) = 0 AND coalesce("/observation.images.up/period_violation_pct", 0) = 0 AND coalesce("/observation.images.side/period_violation_pct", 0) = 0
ORDER BY episode_id
