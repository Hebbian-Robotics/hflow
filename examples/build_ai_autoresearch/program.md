# Bounded training-code research

Improve one training file in an already initialized Build AI CPU reference
experiment. The operator supplies its path and trial command. Read the baseline,
completed development reports, frozen protocol, and current `train.py`. Do not
run initialization, baseline, freeze, confirmation, or export commands.

## Loop

1. Check remaining attempts under `max_trials`. Failed, interrupted, and
   over-budget trial directories count. Stop when the allowance is exhausted.
2. Read development predictions and per-class/per-corpus scores. Write one
   brief, falsifiable hypothesis in `<experiment>/research-notes.md` before editing.
3. Edit only `<experiment>/train.py`. Prefer one conceptual change per trial.
   You can implement sampling/curriculum algorithms, count-preserving training
   augmentations, adapter placement, loss weighting, batching, optimizer logic,
   or learning-rate schedules. Changes need not fit a predeclared parameter grid.
4. Run the supplied `trial-transformers-reference` command once. The runner
   snapshots your source and starts from the same pinned base weights each time.
   Every attempt receives the same wall-clock training allowance on the same CPU.
   Model loading is outside it; source import, setup, training, audit, and saving
   are inside. The parent watchdog kills an over-budget process.
5. Record trial ID, source hash, hypothesis, macro-F1, invalid count, step count,
   runtime, and whether the hypothesis held. Check class scores: a constant answer
   can improve accuracy on imbalanced data without solving the task.
6. Keep a useful change or restore the best trial's source snapshot. Try a new
   hypothesis within the remaining allowance. Stop and summarize evidence and
   failed ideas; let the operator freeze selection and perform confirmation.

## Training contract

Define `train(context: TrainingContext) -> PeftModel`. Use `context.base_model`
for LoRA, `context.samples` for the fixed training records, and `context.batch`
for image loading and prompt/assistant-target masking. You may pass a full-frame,
count-preserving image transform to `context.batch`. Never change teacher labels,
remove hands with crops, or train on development/test images. Report one finite
loss per optimizer update through `context.record_loss`. Honor `context.max_steps`
and stop early enough before `context.remaining_seconds` reaches zero for the
worker to audit/save the adapter (the starter reserves five seconds).

Return LoRA on the supplied base model. You may choose language/vision adapter
placement and which adapter parameters train. Full base-weight, bias, full-module,
and base-architecture changes are rejected. The evaluator reloads the saved LoRA
on fresh base weights in its own process; it never imports your training code.

Do not modify the evaluator, training API, worker, protocol, prompt, dependencies,
reports, snapshots, timers, or checkpoints. Do not inspect source/prepared/test
or confirmation data. Do not increase budgets, install packages, patch runtime
modules, forge results, delete failed attempts, or run concurrent trials.

Optimize fixed three-class development macro-F1. Invalid outputs count as errors.
Teacher labels may be wrong; never relabel to improve a score. Balanced subsets
and exact-frame splits support only limited agreement claims. Do not claim human
ground-truth correctness or recording-independent generalization.
