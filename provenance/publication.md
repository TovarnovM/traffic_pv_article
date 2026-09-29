# Offline publication replay

This component is separate from the working scientific pipeline. It contains saved
scalar observations, selected-model receipts, a portable NumPy/Matplotlib replay,
focused tests, and generated reference results. Training-progress packaging and
other review material belong to the parent integration and are outside this component.

## Inputs and provenance

`publication_export.json` records SHA-256 hashes of the original source, all original
main-series outputs, three raw waves, selection receipts, the combined manifest,
and the exact mathematical function source. The immutable inputs are:

- `sep_research/eval_selection_10m_20260925/test/raw.json`: 25,920 rows;
- `sep_research/eval_selection_10m_20260925/jg_gpu/test/raw.json`: 9,720 rows;
- `sep_research/eval_selection_10m_20260925/jg_gpu/test15/raw.json`: 8,100 rows.

The author-only exporter calls the original `article2/scripts/build_main_series.py`
loader. Before any projection it checks source hashes, local duplicate keys, full-row
identity of 12,960 duplicate control-wave rows, complete coverage, all 15 model
bindings, shared road identities, and B1 equality in every field except fraction.
It then stores 29,700 physical rows. The loader expands the 540 B1 observations over
three fractions to recover 30,780 logical rows (24,300 learned and 6,480 controls).
These B1 aliases are reused observations, never independent replicates.

The gzip JSON preserves exact JSON float values and null laps. Its 25 fields are
listed in `publication_export.json` and `analysis/publication_inputs.py`: row keys,
model IDs, six road-identity fields, four complete metrics, lap time/count,
priority flow/count, vehicle count, measurement steps, status, split and protocol
hash. AV ID arrays, event diagnostics, machine metadata and checkpoint paths are
excluded. Road identities remain on each row and are checked against B1. Every
input file is hash-checked before use; coverage, bindings, metric validity, lap
completion, measurement metadata and flow identities are independently validated.
The gzip stream has timestamp zero and deterministic key/row ordering.

Selection retains every top-level winner decision field except `candidate.path`:
12 winners from `selection_fixed.json`, only the actual two top-level winners
(seeds 20260914/20260916) from `selection_jg_gpu_3seed.json`, and the separate
20260915 winner from `selection_jg_gpu_seed15.json`. Candidate iterations, IDs,
manifest hashes, counters, validation means, rejection decisions, margins and
saved tie-break decisions are retained. This is a winner-receipt replay, not a
rerun of selection or a complete candidate-ranking dataset. The original filename
claiming three seeds does not override its actual top-level winner coverage.
The J_G GPU/non-bitwise-resume provenance is retained explicitly.

## Adaptation and statistical contract

`analysis/publication.py` is adapted from
`sep_research/preliminary_gpu_paper/article2/scripts/build_main_series.py`.
Ten mathematical functions are text-identical; export checks their source equality
and records hashes. The original raw loader and orchestration are replaced by a
compact loader and a small CLI. Table and figure functions take an explicit output
directory. The four needed control style mappings are copied from
`paper/plot_style.py` into `publication_style.py`, avoiding a `paper` namespace
import. Tests derive from the original `test_main_series.py`, with additional
public-input and immutable-reference checks. No license is invented here; license
selection remains with the parent integration.

The calculation preserves equal weights for three training models, sample SD
(ddof=1, not SEM), paired 30-road bootstrap with 10,000 draws and seed 910019,
linear percentiles, and signs from CI endpoints with tolerance 1e-12. Shared
controls are subtracted once. Background screening uses actual B1 road ratios
and requires all three training lines to pass. Lap means remain null when a
required model mean is unavailable; successful-road and paired-success scopes
are unchanged. Figures retain all nine methods, P_L stars, hollow magenta J_G
diamonds, conditional road intervals, and opaque preblended EPS fills.

## Regeneration

Public entry points work from any working directory, using only this component
and installed Python, NumPy and Matplotlib. They do not import the simulator,
models, selection pipeline, or original inputs. In a portable copy:

```sh
python -I /path/to/review/analysis/replay_publication.py
python -I /path/to/review/tests/test_publication.py -v
```

Default output is `review/build/publication`; `--output-dir` selects another
output directory. Six JSON summaries, four LaTeX tables, and five figures in
PDF/SVG/PNG (300 dpi)/EPS are produced, plus a replay-check receipt.
`results/publication/` holds the initial generated references.

All execution for this implementation used the already-running container with
CUDA hidden, without container startup/restart or dependency installation. From
the workspace root, the commands are:

```sh
devcontainer exec --workspace-folder "$PWD" --config "$PWD/sep_research/gpu_duplicate_20260916/runtime/devcontainer.json" env CUDA_VISIBLE_DEVICES= MPLCONFIGDIR=/tmp/publication-matplotlib python -I /workspace/review/_local/export_publication.py
devcontainer exec --workspace-folder "$PWD" --config "$PWD/sep_research/gpu_duplicate_20260916/runtime/devcontainer.json" env CUDA_VISIBLE_DEVICES= MPLCONFIGDIR=/tmp/publication-matplotlib python -I /workspace/review/tests/test_publication.py -v
devcontainer exec --workspace-folder "$PWD" --config "$PWD/sep_research/gpu_duplicate_20260916/runtime/devcontainer.json" env CUDA_VISIBLE_DEVICES= MPLCONFIGDIR=/tmp/publication-matplotlib python -I /workspace/review/analysis/replay_publication.py --output-dir /workspace/review/results/publication
devcontainer exec --workspace-folder "$PWD" --config "$PWD/sep_research/gpu_duplicate_20260916/runtime/devcontainer.json" env CUDA_VISIBLE_DEVICES= MPLCONFIGDIR=/tmp/publication-matplotlib python -I /workspace/review/_local/export_publication.py --check-isolation
```

The `_local` exporter requires the author's original tree. It is not needed for
public reproduction. It refuses to change an existing immutable reference if the
captured reference differs. Public replay never updates that reference.

## Verification performed

Python 3.11.10, NumPy 2.1.2, Matplotlib 3.10.0: all 13 tests passed normally and
again in isolation. Existing Matplotlib/Pyparsing deprecation warnings were
non-fatal. All six canonical payload hashes match the original saved publication
(including all 15 model lines/global means, CI counts, two J_L background-failing
conditions, and logical/physical lap-missing counts). Numerical payload difference
is zero, stronger than the required 1e-12 tolerance. All four tables are
byte-identical. Figure rendering uses the original implementation with only the
output path/import adaptations; all 20 formats were regenerated. The flow figure
was also visually inspected. Figure file bytes are not required to match because
PDF/SVG/EPS metadata and timestamps can vary.

`publication_reference.json` hashes were captured from existing original outputs,
not generated by the replay. Tests detect changed CI signs, duplicated or missing
rows, invalid model bindings, altered shared roads, missing keys, bad lap and flow
metadata, and corrupted compressed inputs. Synthetic tests cover shared-control
subtraction, road coupling, missing laps, denominator failures, SD and CI signs,
and figure methods/legend geometry.

The isolation check copies only analysis, evaluation inputs, selection receipts,
the public test and immutable hash reference into a temporary directory. Fresh
`python -I` children run tests and full replay from an arbitrary temporary working
directory, exercising default output. A Python audit hook denies opens and directory
reads beneath the source workspace, including both immutable original trees; a
probe confirms denial. Project module paths must point into the temporary copy,
and no `sep_research` imports may occur. No generated reference results are copied.
This is an exercised Python access guard, not OS mount isolation. The detailed
receipt is `publication_checks.json`.

The compressed observation file is 1,913,573 bytes (26,220,159 bytes uncompressed).
Metadata plus winner receipts add 23,865 bytes. The full file/size/hash allowlist
is `publication_files.json`; `_local/export_publication.py` is marked author-only.
No checkpoints, pickle files, third-party PDFs, secrets, new simulations, training,
model loading, network calls, commits or changes to the original artifacts are
part of this replay. Git ignore policy, licensing and full-bundle integration are
owned by the parent task.
