# Priority-vehicle control: article reproduction materials

Companion materials for **Separate and joint reinforcement learning for priority-vehicle passage in mixed traffic**, by Mikhail S. Tovarnov and Nikita V. Bykov.

This is a separate, compact package. The working research pipeline and its local archives are not required to replay the published results and have not been moved or replaced. **The package passed 45 checks and clean-export reconstruction. Public licensing and the permanent repository/release URL still require author approval.** See [publication boundaries](docs/PUBLICATION_BOUNDARY.md).

This standalone export contains no parent Git history. For first publication in a new repository and a versioned GitHub Release, see the [Russian publication guide](GITHUB_RELEASE_RU.md). The manuscript PDFs/LaTeX and the Prism editing project are separate from this scientific-materials package.

## What is included

| Directory | Contents |
|---|---|
| `code/` | Frozen scientific sources: simulator, observations, models, training, evaluation and selection; source hashes |
| `configs/` | Physical conditions, training profiles and provenance for five families × three training seeds |
| `data/scenarios/` | Exact scenario seeds, generator settings, 540 road hashes and 1,620 role-assignment hashes |
| `data/evaluation/` | Compact per-road measurements: 29,700 physical observations, expanded to 30,780 logical rows without inventing independent replicates |
| `data/selection/` | Receipts identifying all 15 selected models and their saved selection decisions |
| `data/training/` | 9,165 accepted iteration records in 34 explicitly identified segments; discarded-tail decisions |
| `analysis/` | Offline statistical replay and training-progress plotting |
| `results/publication/` | Six summaries, four manuscript tables and five figures in PDF/SVG/PNG/EPS |
| `results/training/` | PV-speed and team-return learning histories, with resume and selected-checkpoint markers |
| `tests/`, `provenance/` | Focused checks, source/export hashes and verification evidence |

No trained weights, optimizer states, full trajectories, massive logs, third-party articles or cluster credentials are included. Initial-state **arrays** were not found in the finalized artifacts: seeds and hashes are not presented as saved states. The included generator reconstructs a specified case; one low-density case was checked against historical hashes.

## Quick start: reproduce the article's figures and tables

Use Python **3.11** (verified: 3.11.10). Run these commands from this directory, whether it is `review/` in the working repository or the root of a standalone export. A virtual environment is recommended.

```sh
python -m pip install -r requirements-analysis.txt
python -I tests/test_publication.py -v
python -I analysis/replay_publication.py
python -I tests/test_training_progress.py -v
python -I analysis/training_progress.py
```

Outputs go to `build/publication/` and `build/training/`, leaving the shipped reference results untouched. Both plotting programs accept `--output-dir`. No checkpoint, simulator campaign, Ray installation or access to the original research workspace is needed for this route.

The article replay checks immutable reference numerical hashes and exact table bytes. On the verified runtime its statistical payloads match the manuscript, all four tables are byte-identical, and all five PNG figures also match. PDF/SVG/EPS timestamps or rendering metadata can differ; cross-runtime graphic-byte identity is not promised. Inputs are compact raw-measurement projections, **not just previously aggregated means**. [Details and statistical contract](provenance/publication.md).

The training figures show saved rolling **training** episode metrics, not held-out estimates. Segments are plotted separately; there are no confidence bands, invented learner diagnostics or extra smoothing. [Metric definitions and accepted lineage](docs/TRAINING_PROGRESS.md).

## Scientific implementation and initial conditions

The heavier CPU-check environment is separate from offline analysis:

```sh
python -m pip install -r requirements-science.txt
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/traffic-review-numba" \
  python -I tests/test_minimal_acceptance.py
CUDA_VISIBLE_DEVICES= python -I tests/test_scenarios.py
```

The nine acceptance cases exercise observations, synthetic-weight model inference and selected simulator transitions, including an actual compiled longitudinal kernel. They are deliberately bounded, not exhaustive validation. The seed-manifest check does not simulate the full test grid.

See [scientific reproduction](docs/SCIENTIFIC_REPRODUCTION.md) for a single-case initial-condition generator, training-source entrypoint and resource requirements. Repeating training is a separate costly operation, **not** part of package validation. Evaluating the exact published policies additionally requires their omitted weights. The compact condition files do not replace full historical restore contracts.

Historical training used Ray 2.56.0; bounded acceptance here used 2.57.0. CPU/GPU equivalence, identical historical environments and bitwise resume are not claimed. The observed budget is 10,010,624 accepted steps per line; selected models can precede the final iteration. Road bootstrap intervals condition on fixed models and differ from the sample SD across three training lines.

## Integrity and safe package-only export

`RELEASE_FILES.txt` is the explicit public allowlist; `MANIFEST.json` records its files, SHA256 hashes and sizes. Local inventories, author exporters, caches and generated `build/` outputs are excluded.

```sh
python -I tests/test_release.py -v
python -I scripts/release.py verify
python -I scripts/release.py export --destination /path/to/new-empty-export
```

The destination must **not exist**. Export copies only verified files and never deletes local artifacts, initializes Git or includes the parent Git history. Obvious credential checks are heuristic, not a comprehensive security audit. The existing working branch already contains older tracked artifacts: merely changing `.gitignore` does not remove those files or past commits.

For an approved public release, use the reviewed standalone export and a fixed version/tag; then complete `CITATION.cff` with the actual release URL/DOI and chosen license. Do not cite a fabricated DOI or treat the current preparation as a license grant.
