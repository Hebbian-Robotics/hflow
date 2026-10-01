# Bounded recipe search

You are improving the recipe of an already initialized Build AI CPU reference
experiment. The operator supplies its path and trial command. Read the baseline,
completed development reports, and frozen protocol. Do not run initialization,
baseline, freeze, confirmation, or export commands.

## Loop

1. Check remaining attempts under the protocol's `max_trials`. Failed or
   interrupted trial directories count. Stop when the budget is exhausted.
2. Read the latest development predictions and per-class/per-corpus metrics.
   Write one brief hypothesis in `<experiment>/research-notes.md` before editing.
3. Edit only `<experiment>/candidate.json`: learning rate, rank, or sampling
   balance. Prefer changing one value at a time so the hypothesis is interpretable.
   Rank is 2/4/8; sampling is uniform/class-balanced. Keep rate in (0, 0.01].
4. Run the supplied `trial-transformers-reference` command once. Every run starts
   from the same base weights and gets the same optimizer-step budget.
5. Record trial ID, recipe, macro-F1, invalid count, and whether the hypothesis
   held in the notes. Compare against the best completed development result,
   including the untouched baseline. Revert unhelpful recipe changes or try a
   different hypothesis within the remaining budget.
6. Stop and summarize the strongest development result and failed hypotheses.
   Let the operator freeze selection and perform confirmation.

Do not edit the protocol, evaluator, prompt, lockfile, reports, checkpoints,
images, or source data. Do not inspect prepared/test data, confirmation results,
or the operator's filesystem. Do not install alternative dependencies or change
the command/backend. Do not increase the budget or delete failed attempts.

Optimize fixed three-class development macro-F1. Invalid outputs count as
errors. Teacher labels may be wrong; never relabel examples to improve a score.
Class-balanced subsets and exact-frame splits support a limited agreement claim.
Do not claim human-ground-truth accuracy or recording-independent generalization.
