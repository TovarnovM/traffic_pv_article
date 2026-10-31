# Revised article: saved-data reproduction (v1.1.0 candidate)

Companion to **Allocating Reinforcement-Learning Control for Priority-Vehicle Passage in Mixed Traffic**, Mikhail S. Tovarnov and Nikita V. Bykov, Applied Sciences manuscript applsci-4663160.

This is the reproduction package for the revised article: eight compact data tables, numerical verification, regeneration of its **six main and five supplementary figures**, and the selected weights of **21 model configurations / 31 policy states**. No simulation, training, inference, private workspace, checkpoint, or larger author archive is required. The original study's historical materials elsewhere in the repository are not inputs to this package.

**Status:** local v1.1.0 candidate for author review. This preparation does not create or publish a Git tag, GitHub release, DOI, or license grant. [Selected numeric weights](<WEIGHTS.md>) are included in Git as a 35.8 MB archive. The array reader is not a simulator or a restored RLlib training checkpoint; do not describe packaging checks as a new evaluation of the learned policies.

## Quick start

Use Python **3.11** (candidate checks: 3.11.10). The [requirements](<requirements.txt>) pin the independently tested environment: NumPy 2.1.2, pandas 3.0.5 and Matplotlib 3.10.0. The authors' earlier figure build reported Python 3.12, NumPy 2.3.5, pandas 2.2.3 and Matplotlib 3.10.8. Numerical comparisons passed in the candidate environment; cross-version graphical byte identity is not promised. See [verification notes](<VERIFICATION.md>).

From this directory (or the root of its standalone export):

```sh
# Integrity check: Python standard library only.
python3 scripts/release.py verify
python3 scripts/verify_weights.py weights

# Optional environment preparation by the recipient, not performed by the scripts.
python3 -m venv /path/to/new/venv
. /path/to/new/venv/bin/activate
python3 -m pip install -r requirements.txt

# All verification commands read the shipped data without replacing them.
python3 analysis/verify_claims.py data
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 analysis/verify_intervals.py data
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 analysis/verify_flow_intervals.py data
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 analysis/jl_decomposition.py data

# The destination must NOT exist. Produces eleven vector PDFs.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg \
  python3 analysis/plot_figures.py data /absolute/path/to/new/figures

python3 -m unittest discover -s tests -v
```

These commands do not acquire datasets or start a simulator. Verification failures should be investigated, not bypassed by overwriting the reference tables. Installation and computation on a recipient's machine require that machine owner's authorization.

## What the checks establish

- The standard-library claim checker independently reconstructs reported cell means, speed comparisons, lineage flow point estimates/support, threshold failures, combined criteria, selected local values, and lap outcomes.
- The speed-interval checker reconstructs **3678** fixed-cohort PV-speed estimates and both endpoints, with absolute tolerance 1e-10.
- The flow-interval checker reconstructs **3042** main-grid lineage/rule–condition rows: paired road counts, mean roadwise ratios and both percentile endpoints. It needs only the included episodes, not the larger incoming archive.
- The JL diagnostic checks **60 rows** at six densities, including nonlinear ratios and the stopped-time/moving-speed decomposition, recomputing ratios inside the paired bootstrap.
- Plotting consumes the supplied tables; it does not silently recompute a different analysis.

See [methods and data dictionary](<METHODS.md>) and [figure-to-data map](<FIGURE_MAP.md>). JSON checksums and sizes are in [MANIFEST.json](<MANIFEST.json>); [RELEASE_FILES.txt](<RELEASE_FILES.txt>) is the exact export allowlist.

## Data and scope

| File | Statistical unit and contents |
|---|---|
| [episodes.csv.gz](<data/episodes.csv.gz>) | 116,856 model–condition–road observations: PV speed, background flow, stopped fraction, lap count and success-conditional lap time. `observer` identifies 41,760 instrumented rows. |
| [road_keys.csv.gz](<data/road_keys.csv.gz>) | 900 density–road-index identities with scenario/state hashes, PV identity, warm-up/measurement seeds and horizon. Simulation identifiers, not recorded real drivers. |
| [global_points.csv.gz](<data/global_points.csv.gz>) | Archived global means, paired contrasts, intervals and support; `view` separates estimands. |
| [lineage_flow.csv](<data/lineage_flow.csv>) | 3042 main-grid individual flow ratios and their fixed-policy percentile intervals; diagnostic D2_L excluded. |
| [JL_substitution.csv](<data/JL_substitution.csv>) | Six all-AV densities; three fixed coordinators, native versus learned PV, 22 common roads, speed/stop decomposition and intervals. |
| [local_points.csv](<data/local_points.csv>) | Model-averaged moving/stationary-window comparisons and restart profiles at density .40 / AV fraction .50, with event/frame and road support. |
| [local_lineages.csv.gz](<data/local_lineages.csv.gz>) | Individual-model local means, between-road SDs and support. Used to reconstruct model-level summaries and the Figure 6 range. |
| [training.csv.gz](<data/training.csv.gz>) | 11,609 accepted iteration positions from 19 models, including gaps, segments and selected-checkpoint flags. No histories are invented for the two NoMP models. |

### Reproduction boundary

Global statistics can be recomputed from the included per-road observations. Local plotted profiles can be reconstructed and checked from the included saved aggregates; **raw local trajectories and per-event source records are not included**, so this is not independent re-extraction of the local observer from traffic traces. The training file contains retained metrics, not optimizer or environment states. State hashes are not saved physical-state arrays.

The selected numeric weights are included; optimizer states, full trajectories, temporary files, working logs, reviewer correspondence, manuscript drafts, marked copies, and additional exploratory figures are not. Re-evaluating the policies requires the matching scientific implementation and a separate computational protocol; reproducing training additionally requires its training setup. The full repository provides the corresponding model-source references; the standalone export supplies numeric arrays, not a complete simulation launcher. New local metrics outside these tables require the larger source data.

## Export only this release candidate

```sh
python3 scripts/release.py export --destination /absolute/path/to/new-export
```

The destination must not exist. Only the allowlisted, checksum-verified files are copied; local outputs and unrelated repository files are excluded. Use **this directory's exporter**, not the root-level historical export list, for the v1.1.0 asset. The exporter performs limited credential-pattern checks; those are not a general security certification. It neither runs Git nor publishes anything. Generated PDFs and local check logs are deliberately not shipped.

## Citation and licensing

[CITATION.cff](<CITATION.cff>) identifies the revised article and intended version. No DOI or release date is invented. The versioned public link can be inserted into the manuscript only after the actual release is published.

This candidate does not assign a new license or imply author approval of a license grant. Existing applicable rights/notices remain unchanged; authors must confirm public distribution and licensing before publication.
