# Reduced T1/T2 acceptance — committed criteria before first execution

User explicitly supersedes the old full matrix with representative minimal acceptance.
PASS requires all nine named unittest tests below, no skip/fallback, and successful
same-process frozen-origin/hash gates before and after tests. A failed required check
is not removed or loosened to manufacture PASS. Scientific source is immutable.

Allowlist (prefix `__main__.MinimalAcceptance.`):

1. `test_t1_pv48_manual_wrap_missing`: exact float32 48-vector for five vehicles,
   x=199/0 wrap, distance 60 inclusive / 61 absent, missing lane/neighbors,
   cooldown stay-only mask; input state unchanged.
2. `test_t1_local23812`: synthetic CPU Torch forward vs NumPy-input/NumPy-weight
   inference adapter, counts/shapes/logits/value/masked argmax; one AV and zero AV.
3. `test_t1_graph411012`: same, graph padding perturbation invariance.
4. `test_t1_av355460`: same, padded stay masks and graph padding invariance.
   In 2–4: expected logits/value tolerance atol=rtol=1e-6; masks/actions exact;
   observations and Torch RNG unchanged. No backward, SGD, trained checkpoint.
   This is Torch forward versus the repository's Torch inference adapter accepting
   NumPy inputs/weights, NOT an independent NumPy numerical implementation.
5. `test_t2_phase_occupancy_ids`: two vehicles, explicit LC then longitudinal
   versus full step; manually expected first vehicle lane=1, v=4, x=14;
   occupancy, IDs/alive, input purity, exact final RNG and fields.
6. `test_t2_sparse_applied_rejected`: one applied PV request and two-role conflict,
   one physical transition per case, event counts, rejection reasons, reward and
   no native lane change following rejected conflict.
7. `test_t2_native_vs_explicit_stay`: blocked PV fixture; absent request equals
   native simulator (state/RNG) and permits LC, explicit zero suppresses LC.
8. `test_t2_b2_c1`: odd-ID equal-clearance tie yields right; one real applied
   request, wrapper bypasses its separate cooldown; controller then suppresses
   exactly one decision before reissue. Decisions reuse fixed trigger state,
   so this is controller semantics plus one integration step, not trajectory proof.
9. `test_t2_reference_optimized_compiled`: three unit-length full steps from one
   four-vehicle wraparound state with identical seed=123; p_lane_change=0;
   exact every-state-field and RNG parity, invariants/IDs. Instrument the Python
   dispatch boundary to call the original Numba CPUDispatcher; require three
   calls and nonempty nopython signatures. No skip or reference fallback PASS.

Budget: nine tests, one CPU test process, one Torch intra/inter-op thread; no
warmup/cache construction, no Ray cluster, no training/evaluation campaign.
At most 40 counted simulator/phase calls (planned 15: 3 phase/full + 2 sparse +
3 native/stay/oracle + 1 B2 + 6 backend). Phase calls count conservatively.
Zero transitions in observation/model checks. Simulator fixtures have 2–4 vehicles;
PV vector has 5; model fixtures 3. Environment objects are snapshot-installed,
never randomly reset. Torch seed 731; measurement 15; phase 0; backend 123.
Snapshot metadata seeds 11/12/13/14 describe synthetic installation only, with
warmup_steps=0. No historical sampled-road provenance is implied.

Isolation: python -I, explicit code/src and code at front, remove workspace sys.path
entries; no pytest/conftest/plugins/current test imports. Record every loaded
sep_research/snfs_traffic module's __file__ and SHA256 in the same process.
science_sources.json is generated before execution from frozen files and checked
against all 15 contract.sources for listed modules (package initializer is extra).
The runtime checks hashes before importing and census before/after suite; archive
config hashes, script/criteria hashes, versions, exact command/stdout/XML/JSON.
CPU run_config files are derivative adapters; this is new acceptance on frozen
source, NOT proof of historical test import origins.

Portable command (set REVIEW to the absolute extracted review directory):

```sh
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -I "$REVIEW/tests/test_minimal_acceptance.py"
```

Reports go to build/verification; inputs are provenance/science_sources.json.
Excluded: exhaustive stochastic branches, optimized sparse overrides and compiled
lane changes with nonzero probability, CPU/GPU equivalence, historical import
proof, checkpoint loading, bitwise resume, training or scientific reevaluation.
