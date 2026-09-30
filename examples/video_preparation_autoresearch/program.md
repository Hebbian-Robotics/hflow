# Autoresearch: a model-input preparation budget

Edit only `examples/video_preparation_autoresearch/candidate.py`.
The evaluator, synthetic source generator, motion detector, encoding settings,
development seeds and confirmation seeds are fixed. Do not inspect confirmation
results during development. The commands below are run from the repository root.

1. Run the baseline with a fresh output directory:

   ```bash
   uv run --locked --project examples/video_preparation_autoresearch \
     python examples/video_preparation_autoresearch/evaluate.py evaluate \
     --output data/preparation-search/baseline
   ```

2. Propose one change to frame rate or resolution and explain the hypothesis.
   Keep duration, source offset and all other preparation settings fixed.
   Evaluate into another fresh directory with the same command.

3. Keep a candidate only if development accuracy is 1.0 and `pixel_frame_work`
   is lower than the incumbent. Unassessable predictions count as incorrect.
   Revert rejected changes. Record the hypothesis and report path for every
   attempt in your own experiment log. Limit this example to eight attempts.

4. Select one candidate, stop editing, and freeze its development report:

   ```bash
   uv run --locked --project examples/video_preparation_autoresearch \
     python examples/video_preparation_autoresearch/evaluate.py freeze \
     data/preparation-search/selected/report.json \
     --output data/preparation-search/selection.json
   ```

5. Evaluate the frozen budget on the separate confirmation episodes:

   ```bash
   uv run --locked --project examples/video_preparation_autoresearch \
     python examples/video_preparation_autoresearch/evaluate.py confirm \
     data/preparation-search/selection.json \
     data/preparation-search/selected/report.json \
     --output data/preparation-search/confirmation
   ```

Report both development and confirmation accuracy, decoded frame counts,
pixel-frame work, and preparation time. Do not retune on these confirmation
episodes; they become exposed after this run. A new research cycle needs new
independent confirmation inputs.

`pixel_frame_work` is an input-size proxy, not model token count, FLOPs, GPU cost
or latency. Preparation timing is a single CPU pass and includes startup effects.
This synthetic example validates the mechanics; it cannot support claims about
VLM quality or real-world robot behavior. Your coding agent supplies the loop;
HFlow supplies data preparation. No agent service or training engine is built
into HFlow.
