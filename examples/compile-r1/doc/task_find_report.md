# Task Find Report

## Suitability Judgment

Current `compile-r1` / AceCode coding-function task is **not** a good primary source for tool-use cold start under `task_find.md`.

Reasons:
- AceCode total rows: 87149
- Explicit table-like rows: 2313
- Finance-like rows: 956
- Scale-like rows: 555
- Pure code-task phrasing rows: 67903
- The dominant distribution is still "natural-language function spec -> Python implementation", not table/numeric/scale reasoning.
- That means no-tool can often solve the task directly, so it is hard to create a clean `no-tool wrong / tool right` signal.

The new dataset route in `task_find.md` is appropriate because TabMWP / FinQA / TAT-QA / TabularGSM are natively numeric, table-based, scale-sensitive, or hybrid table+text reasoning tasks.

## Output Counts

- normalized `tabmwp`: 38431
- normalized `finqa`: 8281
- normalized `tatqa`: 16546
- normalized `tabulargsm`: 3404
- filtered `tabmwp_candidates`: 7922
- filtered `finqa_candidates`: 5876
- filtered `tatqa_candidates`: 5762
- filtered `eval_tabulargsm_hard`: 797
- filtered `eval_tabulargsm_robust`: 1000

## Notes

- MATH official repo clone does not contain raw train/test JSON locally; skipped optional migration normalization for now.