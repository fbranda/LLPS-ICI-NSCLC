# LLPS-associated transcriptomic signatures in ICI-treated NSCLC: multicohort reanalysis

**Reproducibility repository for a scientific manuscript in preparation for the Computational and Structural Biotechnology Journal (CSBJ).** Manuscript status: **in preparation / not yet accepted or published**. No article DOI or published GitHub URL should be inferred from this archive.

This repository contains the **actual processed public data, human PhaSePro curations, Python source code, analysis-level output tables, locked predictions, and sensitivity results** underlying an exploratory retrospective multicohort reanalysis of LLPS-associated transcripts in immune checkpoint inhibitor (ICI)-treated non-small-cell lung cancer.

The central question is **cross-cohort reproducibility and the incremental association with ICI-treated outcomes**, *not* a claim to have validated an LLPS-specific treatment-benefit biomarker or to have directly measured liquid-liquid phase separation.

## Main audited results

| Item | Analysis finding | Interpretation |
|---|---|---|
| Curated LLPS-related set | 59 human PhaSePro genes; 22 shared response features | Partial assay coverage |
| Development response cohort | GSE161537: 76 labeled, 20 responders | Targeted expression assay |
| External response cohort | GSE126044: 16 labeled, 5 responders | Very small, distinct assay |
| 22-gene nested-CV model | Mean AUROC 0.415 across 3 repeats | Weak internal discrimination |
| Locked external AUROC | 0.836; conditional exact one-sided p = 0.019 | Statistically unusual ranking under fixed-label permutation, *not* validation |
| HOT immune score benchmark | AUROC ~0.658 vs HOT+LLPS ~0.477 | No observed incremental discrimination |
| Assay-matched random panels | 65.5% of 200 panels at least as good as fixed LLPS panel | Exploratory specificity check |
| GEO-linked early PFS | France3 n=26 + France4 n=70; MED1 HR 0.798, p=0.083 | No FDR-confirmed associations |

The external 0.836 AUROC is based on **only five responders**, and contrasts with internal performance. Neither this result nor any bulk RNA association demonstrates ICI-specific treatment benefit: all cohorts are treated-only and lack a comparator arm. Data limitations and incomplete original assay coverage are documented in [`data/README.md`](data/README.md).

## Repository map

```
analysis/                  Original and post-review analysis code
analysis/results/          Archived baseline analysis outputs, CSV/JSON and figures
analysis/corrected_results/ Post-review sensitivity outputs and figures
data/raw/                  Public, processed GEO matrices + PhaSePro source dump
data/clinical_extraction/  Tables derived from public GEO source metadata
gene_sets/                 PhaSePro human gene set and provenance
figures/manuscript/        Formatted figures corresponding to the journal draft
scripts/                   Integrity checks for input files and output summaries
requirements.txt           Python dependency versions for computational replay
run_all.sh                 Re-executes the numerical analyses
SHA256_INPUTS.txt          File-level integrity hashes for source inputs
```

## Reproducing the analysis (Linux, macOS, or WSL)

Requires Python 3.11 or later and ~1 GB of free working space (including intermediate outputs); all raw public inputs are included in this archive, so no network data fetch is needed for replay.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check_inputs.py
bash run_all.sh
python scripts/check_outputs.py
```

`run_all.sh` does **not** invoke a network API or install packages. It runs original and corrective scripts and regenerates derived results locally. Runtime depends on computer and library versions. The scripts set single-threaded BLAS/OMP environments to limit computational variability. For archival traceability, source inputs are recorded with SHA-256 checksums. Some statistical results can vary slightly across scientific package versions; use the exact requirements versions where possible.

## Main scientific limitations

1. PhaSePro membership and mRNA levels are **not** direct measures of phase behavior or condensates in tumor specimens.
2. Not all proteins or previously published LLPS signature components were measured; the study does **not** directly validate published LLPSAS/MLORS signatures.
3. Unharmonized treatment outcomes, different assays and tiny external validation cohort limit extrapolation.
4. The HOT score was previously analyzed in a related source cohort; this is not an independent biomarker trial.
5. France3 sample linkage and France4 apparently truncated expression file limit PFS interpretation.
6. No biomarker-by-treatment interaction can be estimated from treated-only cohorts.

## Source attribution

The source GEO data were generated and released by the respective original research teams. Please cite GSE161537, GSE126044, GSE190265, GSE190266, GSE135222 and the original PhaSePro publication (Mészáros et al., *Nucleic Acids Research*, 2020, https://doi.org/10.1093/nar/gkz848). This archive does not assert ownership of those datasets. The code is made available under the MIT license; **third-party data retain their original applicable terms**.

## Citing this code

When the paper is published, cite the paper and the public repository at its *actual, verified* GitHub URL (prefer a fixed commit hash or DOI-backed archive). There is no valid public repository URL until the files have been uploaded to a repository.

## Editorial transparency

This code and data archive provides the basis for independent numerical verification. It is **not** a claim that the raw GEO files were independently resequenced, that all original clinical data are verified, or that the model is clinically deployable. Any manuscript code-availability statement should link only to the repository after it is public and verified.
