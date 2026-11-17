#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
mkdir -p analysis/data
ln -sfn ../../data/raw analysis/data/raw
ln -sfn ../../data/clinical_extraction analysis/data/clinical_extraction
ln -sfn ../gene_sets analysis/gene_sets
python3 analysis/run_real_analyses.py
python3 analysis/run_sensitivity.py
python3 analysis/run_model_benchmarks.py
python3 analysis/make_additional_figs.py
python3 analysis/run_corrective_analyses.py
printf 'Analysis complete. Baseline and corrected outputs in analysis/results and analysis/corrected_results.\n'
