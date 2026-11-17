# Traceability and replay scope

- Main analysis in `analysis/run_real_analyses.py`: ROC, fivefold repeated nested CV, gene-level response analyses, 6-month PFS association and audit tables.
- Post-review script `analysis/run_corrective_analyses.py`: 4,368 external conditional label permutations, LAT immune marker adjustment, immune HOT comparator with mandatory HOT feature, expression-matched reference panels (200) and 26+70 GEO-linked PFS analyses.
- The post-review comparisons are **sensitivity analyses**, not prospective prespecified hypothesis tests.
- The baseline and post-review results both remain in the archive to preserve the history of the sensitivity analyses; for final early-PFS estimates follow the link-verified subset.
- Independently audit `analysis/corrected_results/` to confirm the external conditional p-value and that the correct clinical denominators were used.
- Do not interpret the exact permutation p-value as a correction for overall model-selection multiplicity or model-building uncertainty.
- No raw data from patients newly recruited by the authors are included; all input matrices are from public GEO sources.
- The journal manuscript itself is an editorially revised Word document and is **not automatically produced by the scripts**. The analysis outputs underpin the quantitative claims.
