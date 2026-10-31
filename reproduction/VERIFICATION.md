# Candidate verification scope

## Performed

The v1.1.0 candidate was checked in a separate CPU-only runtime with **Python 3.11.10, NumPy 2.1.2, pandas 3.0.5 and Matplotlib 3.10.0**, using one numerical thread and no network or dependency installation.

- **15 regression tests passed**: ratio aggregation/common-road selection, interval-table keys/counts/endpoints, JL decomposition/output guards, and selected-weights archive validation.
- [Claim checker](<analysis/verify_claims.py>): all supplied numerical assertions passed.
- [Speed interval verification](<analysis/verify_intervals.py>): 3678 estimates and both endpoints reproduced with absolute tolerance 1e-10.
- [Individual flow verification](<analysis/verify_flow_intervals.py>): all 3042 estimates, paired supports and both endpoints reproduced with absolute tolerance 1e-10.
- [JL decomposition](<analysis/jl_decomposition.py>): all 60 rows and interval endpoints reproduced with absolute tolerance 1e-10.
- [Figure reconstruction](<analysis/plot_figures.py>): all eleven expected PDFs generated successfully in a fresh directory.
- [Weights verifier](<scripts/verify_weights.py>): all 21 selected configurations / 31 numeric policy states passed; each NPZ is byte-identical to the selected original, with SHA256 checked against its original manifest when packaged.
- [Numeric reader](<policies/weights.py>): all 31 states loaded using NumPy with `allow_pickle=False`; exact parameter-key sets, shapes, float32 dtype and finiteness checked. The aggregate parameter count is 9,586,012.

The first regression run exposed a test-fixture error: a supposedly changed upper endpoint equaled its original value. The fixture was corrected, without changing the statistical estimator, and the complete run then passed. Temporary outputs and runtime logs are not release contents.

## Not claimed

These checks did not construct Torch models, run policy inference, step a simulator, rerun Monte Carlo evaluation, retrain policies, restore optimizer/environment state, or establish CPU/GPU action equivalence. The weights themselves were copied unchanged, not converted or quantized. A forward pass requires the correct model and observation implementation at the same repository revision.

Local profiles were checked/replotted from saved aggregates, not re-extracted from raw trajectories. The regenerated PDFs were counted and produced without plotting errors; byte identity with PDFs generated under the authors' earlier plotting environment is not required.

## Repeating checks

Commands and exact export/integrity rules are in [README](<README.md>), [weights guide](<WEIGHTS.md>) and [MANIFEST.json](<MANIFEST.json>). Numerical inputs are not overwritten. The package's public manifest and member hashes provide integrity checks, not independent proof of scientific validity or authenticity of an untrusted source.
