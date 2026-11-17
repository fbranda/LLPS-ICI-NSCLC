# Computational analyses

Scripts are executed from the repository root using `bash run_all.sh`. Source code expects `analysis/data` and `analysis/gene_sets` links to the repository's input folders; `run_all.sh` creates these automatically.

| Script | Role |
|---|---|
| `run_real_analyses.py` | Original cohort audit, single-gene analyses, ranked 22-gene L2 logistic model, nested CV, locked external test and PFS meta-analysis |
| `run_sensitivity.py` | Original sensitivity checks |
| `run_model_benchmarks.py` | Secondary RF/gradient-boosting algorithms |
| `make_additional_figs.py` | Additional figures |
| `run_corrective_analyses.py` | Post-review exact 4,368 label permutations, LAT immune association, HOT forced-comparator analysis, 200 matched-gene random panels and GEO-linked PFS sensitivity |

Re-analysis from scratch regenerates `analysis/results/` and `analysis/corrected_results/`. The currently committed results are the archived baseline outputs. Analyses labeled *post-review* or *exploratory* were not preregistered.

Do not use `analysis/results/PFS_6month_cox_fixed_effect_meta.csv` as the **primary** PFS result: the paper's audit-focused primary PFS estimate is the GEO-linked subset from `analysis/corrected_results/PFS_GEO_linked_26_plus_70_genes.csv`.
