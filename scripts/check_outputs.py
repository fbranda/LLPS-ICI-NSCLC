#!/usr/bin/env python3
"""Compare core reproducible numerical findings against the paper archive."""
from pathlib import Path
import math, json, csv
root=Path(__file__).resolve().parents[1]
a=root/'analysis'
main=json.loads((a/'results/response_machine_learning.json').read_text())
corrected=json.loads((a/'corrected_results/CORRECTIVE_SUMMARY.json').read_text())

assert main['development_n']==76 and main['development_response_n']==20
assert main['external_n']==16 and main['external_response_n']==5
assert len(main['shared_genes'])==22
assert math.isclose(main['mean_nested_AUC'],.4148809523809524,abs_tol=1e-6)
assert math.isclose(main['external_AUC'],.8363636363636364,abs_tol=1e-6)
assert corrected['external_exact_permutation']['n_exact_label_assignments']==4368
assert math.isclose(corrected['external_exact_permutation']['one_sided_label_permutation_p'],.0190018315018315,abs_tol=1e-6)
assert math.isclose(corrected['LAT_immune_proxy']['LAT_Tcell_spearman_rho'],.5426384142173615,abs_tol=1e-6)
assert math.isclose(corrected['random_panel_assay_benchmark']['fraction_random_panels_AUROC_at_least_LLPS'],.655,abs_tol=1e-6)
assert math.isclose(corrected['GEO_linked_PFS']['MED1_row'][0]['HR_pooled'],.7976176789101239,abs_tol=1e-6)
print('OUTPUT AUDIT PASSED: principal archived model and PFS summary statistics match.')
