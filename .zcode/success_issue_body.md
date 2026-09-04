## Current behavior

[docs/FORMAT.md:109](https://github.com/Hebbian-Robotics/hflow/blob/main/docs/FORMAT.md#L109) defines `episode/v1`'s `success` as "Collector-labeled outcome, `"true"`/`"false"`", and [:119](https://github.com/Hebbian-Robotics/hflow/blob/main/docs/FORMAT.md#L119) states the record is "copied/merged from the source recording". The canonical transform honors that: it copies `episode/v1` through from the source untouched ([src/hflow/transform.py:786-787](https://github.com/Hebbian-Robotics/hflow/blob/main/src/hflow/transform.py#L786-L787)).

The LeRobot importer does not. It constructs the record from scratch ([src/hflow/importers/lerobot.py:1030-1041](https://github.com/Hebbian-Robotics/hflow/blob/main/src/hflow/importers/lerobot.py#L1030-L1041)) and hardcodes `"success": "true"` at [:1035](https://github.com/Hebbian-Robotics/hflow/blob/main/src/hflow/importers/lerobot.py#L1035) for every episode it converts. Its episodes query ([src/hflow/importers/lerobot.py:487](https://github.com/Hebbian-Robotics/hflow/blob/main/src/hflow/importers/lerobot.py#L487)) reads seven columns and never touches an outcome field.

The outcome signal it skips exists in the format and in the wild. LeRobot v3 carries `next.success` as a declared per-frame feature, and the episodes parquet aggregates it per episode as `stats/next.success/min|max`. `lerobot/pusht` — the reference dataset for the v3 docs — declares it in `meta/info.json` and carries the aggregates for all 206 episodes; in the current revision every frame of every episode reads `False`.

The label then becomes a queryable fact: `success` is one of the four episode keys promoted to first-class catalog columns ([src/hflow/catalog.py:59](https://github.com/Hebbian-Robotics/hflow/blob/main/src/hflow/catalog.py#L59), [:68](https://github.com/Hebbian-Robotics/hflow/blob/main/src/hflow/catalog.py#L68), promoted at [:863](https://github.com/Hebbian-Robotics/hflow/blob/main/src/hflow/catalog.py#L863)).

## Controlled result

Ran the real import path (network and ffmpeg stubbed), read the delivery back through the real `Episode` reader, appended through the real `Catalog.append_episode`, and queried with real SQL:

```
--- stage 1: what the source declares ---
  data parquet frame (episode 0): next.success = False
  episodes parquet aggregate (episode 0): stats/next.success/max = ['False']
--- stage 2: what the importer shipped ---
  lerobot_episode_0001.mcap: episode/v1 success = 'true'
--- stage 3: the buyer's query ---
  SELECT ... WHERE success = 'true'  ->  2 row(s):
    episode_id=ba19d7fb94cdba01  task='task-1'  success='true'
    episode_id=e9e1a9ad1306d39d  task='task-0'  success='true'
  (catalog holds 2 episode(s) in total)
```

A demo whose collector label reads never-succeeded is delivered as `success: "true"` and returned by the exact filter a buyer would run. On `lerobot/pusht` at scale, the same disagreement holds for all 206 episodes: hflow's column says `true` where the source says `False`, 100 percent of the time.

## Why now

Third brick of the thread #376 and #379 opened: #376 removed a stamp asserting a measurement nobody took, #389 gave the delivery a checkable receipt. This is the last record in the family that speaks without evidence, and unlike the first two it speaks about content: which demos worked. That is the column a data buyer filters on.

## What to build

Open call, not assumed:

1. **Read the label**: when the source carries an outcome feature (`next.success` or equivalent), derive the episode outcome and stamp it. Whether episode success is any-frame or last-frame is a methodology choice to name.
2. **Stamp unlabeled**: when the source carries no outcome feature, omit the key. FORMAT.md:119 already says all keys are optional; omitting is the only encoding that cannot be mistaken for a collector's judgment.
3. **Column semantics**: how the catalog column represents unlabeled episodes (NULL versus a value) is the catalog's decision to make once, deliberately.

Combinations are plausible: read when present, omit when absent, document both.

## Definition of done

1. The importer never invents a collector label: the episode record is copied from the source or omits the key.
2. When the source declares an outcome feature, the delivered record reports it, with the derivation named.
3. The catalog `success` column carries real information or NULL, pinned by tests through the real import and catalog path.
4. The transform's copy path is untouched; non-LeRobot imports are unaffected.
5. The fixture above lands as a test.

## Non-goals

- Changing the transform's episode/v1 copy path
- Judging whether pusht's upstream conversion should have populated `next.success`; that is upstream's question
- Changing catalog dedupe or the promoted-keys mechanism

## Validation

```bash
uv run ruff check --fix
uv run ruff format
uv run ty check
uv run pytest -q tests/test_lerobot_converter.py tests/test_catalog.py
uv run pytest -q
```
