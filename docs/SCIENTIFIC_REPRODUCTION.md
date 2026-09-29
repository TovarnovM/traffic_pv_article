# Scientific source, conditions and bounded acceptance

This slice contains the actual frozen simulator, observations, models, inference,
training, checkpoint/continuation and evaluation/selection sources. Scientific
source hashes and original-to-published hashes are in
`provenance/science_sources.json`. The numerical source copies are byte-identical.
Training and selection provenance helpers remain for review of lifecycle and
validation contracts. No SSH transport, deployment, artifact collector, private
runtime inspection, trained weights, optimizer binaries, or training logs are
included. No licence is asserted by this slice.

`configs/training_lines.json` identifies all five families (A, P_L, P_G, J_L,
J_G), each with seeds 20260914–20260916. Its compact projections retain environment,
model, PPO, resource and continuation values, with original config hashes. They
are readable conditions, not replacements for full RLlib restore contracts.
CPU configs are derivative evaluation adapters; they do not establish historical
training environments or import origins. GPU configs originate in training.

Each line targets 611 batches of 16384 accepted environment steps: **10010624**,
rounded up from the nominal 10000000. Candidates are 450, 500, 550, 600 and 611.
Training cycles through random permutations of twelve conditions: densities
0.1/0.2/0.3/0.4 crossed with AV fractions 0.90/0.95/1.00. Episodes last 1000 steps;
1000-step all-HDV warmup states are cached by density, not generated freshly on
every reset. See `src/snfs_traffic/rl/centralized_pv_env.py` for cache and RNG ownership.

`configs/evaluation.json` records the physical conditions. Validation uses density
0.30, AV fraction 1.00 and twenty roads from root 610003. Only the leading two
admissible candidates receive fifty fresh roads from root 620000 when their speed
gap is strictly below 0.05. The rejection guard, all-flagged fallback, final
speed/earlier-iteration tie rule and pairing are preserved in
`sep_research/eval_selection_10m/selection.py` and `tiebreak.py`; J_G joins through
`jg_gpu.py`. Test conditions are eighteen densities 0.05 through 0.90 in steps of
0.05, three AV fractions, thirty roads each, root 710009, and 2000 measurement steps.

## Commands independent of the current directory

Set `REVIEW` to the absolute location of the extracted `review` directory, using
an existing Python environment with `requirements-science.txt` dependencies.
The commands work from any directory. `-I` is required; do not replace it with
editable-package imports. Each bounded check uses CPU only.

```sh
export REVIEW="$(realpath path/to/review)"
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/traffic-science-numba" \
  python -I "$REVIEW/tests/test_minimal_acceptance.py"
CUDA_VISIBLE_DEVICES= python -I "$REVIEW/tests/test_scenarios.py"
```

Acceptance writes `build/verification/summary.json` and `results.xml`; these are
new outputs, never golden inputs. The pre-run scope is
`provenance/science_acceptance_criteria.md`. All nine tests must pass, with no
skip or fallback. Same-process origin and hash checks run before and after the
suite. The source manifest covers more than the exercised modules. Tests use
synthetic weights only, with no checkpoint, backward pass, SGD or Ray cluster.
The original seeds, exact assertions and tolerances are preserved.

## Initial conditions

The historical evaluator generates states in memory; no serialized warmed arrays
were located in the inspected finalized campaigns. `data/scenarios/seeds.json`
contains 1620 seed rows; `hashes.json` contains 540 road records and 1620 AV role
assignment hashes. Fractions share the same warm road and PV, and use nested AV
subsets. Large repeated AV-ID arrays are omitted; the ordered subset hash remains.
Hashes attest historical arrays but cannot reconstruct them by themselves.

The small CLI calls the frozen `evaluate.prepare_group` helper. It requires an
explicit density, repetition and new output directory; it never defaults to a
540-road campaign. For a bounded verification of the first low-density road:

```sh
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/traffic-science-numba" \
  timeout 120s python -I "$REVIEW/scripts/prepare_scenarios.py" \
  --density-token 5 --run-index 0 --out-dir "$REVIEW/build/verification/scenario_5_0"
```

The command performs exactly one 1000-step warmup, no measurement rollout and
no model loading. It checks physical, warmed, pre-AV, PV-role, activated-role and
ordered AV-subset hashes before publishing numeric NPZ arrays and a JSON report.
Choose a new output directory on rerun. A mismatch fails; it is not silently
accepted as historical reproduction. Other cases are intentionally opt-in and
can be expensive. Seeds alone are not a historical restore contract: frozen
parameters, code, RNG/backend behavior and assignment ordering also matter.

## Training, evaluation and publication replay

Published-statistic replay is handled by the separate `analysis` slice and
`requirements-analysis.txt`; it uses saved numeric results and needs no models.
The minimal tests and initial-state generator do not rerun the paper's learning
or full evaluation. To inspect the original trainer's supported arguments without
starting training, use the source tree explicitly:

```sh
CUDA_VISIBLE_DEVICES= python -I -B -c \
 'import runpy,sys; from pathlib import Path; p=Path(sys.argv.pop(1)); sys.path[:0]=[str(p/"src"),str(p)]; runpy.run_module("sep_research.train",run_name="__main__")' \
 "$REVIEW/code" --help
```

The trainer's default main budget and resource/minibatch choices are older than
the 10M campaign. A new run must explicitly use the corresponding line's compact
conditions and original CLI flags, including the 10000000 budget, 611 iterations,
16384 batch size, 256 minibatch, 6 epochs and line-specific worker/device resources.
Such a run is new training, not a recovered historical continuation. The source
retains strict contract checks; compact configs cannot be passed as restored
checkpoint contracts. Full evaluation requires paired selected model bundles,
full original contracts and receipt provenance. Historical operational admission
and archive gates are not a self-contained public campaign launcher. They must
not be bypassed to claim historical reproduction. Transport and deployment are
outside this portable slice. Unchanged historical gate sources retain their
host/registry labels as contract constants; these are not credentials or a
promise that the old institutional deployment is publicly available. Review
those labels before release if institutional naming is considered private.

Historical continuation restores learner/optimizer state but does not establish
bitwise environment/cache/episode/RNG replay. CPU P_G seed 20260914 crosses an
interrupted segment, so earlier candidates can belong to predecessor attempts.
GPU resume segments and changed resources are disclosed per line. No trained
weights are provided or loaded here. CPU/GPU equivalence, historical import
origins and exact historical resumability are not claimed.

Verified acceptance runtime: Python 3.11.10, NumPy 2.1.2, Torch 2.5.1+cu118 on CPU,
Ray 2.57.0, Numba 0.67.0, Gymnasium 1.2.2. Historical config targets specify Ray
2.56.0; the acceptance environment is distinct. The nine tests cover representative
cases, not exhaustive stochastic branches, optimized sparse overrides, or
compiled lane-change equivalence at nonzero lane-change probability. B2 uses
one physical step followed by fixed-state cooldown decisions. Network parity
compares two Torch paths, not an independent NumPy implementation.

For example, the following explicitly starts **costly new A-family CPU training**
with the saved batch/resource geometry. It was not executed during packaging and
is not equivalent to the original segmented run; use a fresh output directory.
Historical resource limits and versions require separate provisioning.

```sh
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 python -I -B -c \
 'import runpy,sys; from pathlib import Path; p=Path(sys.argv.pop(1)); sys.path[:0]=[str(p/"src"),str(p)]; runpy.run_module("sep_research.train",run_name="__main__")' \
 "$REVIEW/code" --family A --seed 20260914 --profile main \
 --workers 16 --num-cpus-per-env-runner 1 --ray-num-cpus 36 --num-gpus 0 \
 --object-store-memory 2147483648 --train-batch-size 16384 --minibatch-size 256 \
 --rollout-fragment-length 256 --epochs 6 --budget 10000000 --iterations 611 \
 --out-dir "$REVIEW/build/new_A_20260914"
```

The frozen optional legacy export route imports `research.nograph`, which is
absent from the frozen snapshot itself. It is not used by the five-family tests,
initial-condition generator or current training entrypoint. Legacy binary export
is therefore not a supported self-contained operation in this slice. No fallback
from the mutable development workspace was added.

Only files listed in `provenance/science_bundle.json` are publication inputs.
Generated bytecode caches and `build/verification` are runtime outputs, excluded
from that inventory; packaging preserved them in place. The recorded trainer
help check produced bytecode before `-B` was added to the documented command.
