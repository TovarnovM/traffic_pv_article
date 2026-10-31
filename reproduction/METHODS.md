# Saved-data contract

## Conditions and identities

`d` and `f` are integer percentage tokens: `d=40` means density .40; `f=95` means 95% of background vehicles are AVs. Test densities are .05,.10,…,.90; AV fractions are 10,25,50,75,90,95,100%. The simulated ring has four lanes, 1000 cells per lane, one one-cell PV, 1000 warm-up transitions and 2000 measured transitions. Units are uncalibrated lattice units, not real seconds/metres. PV/background speed caps are 6/5 cells per step.

A learns the AV coordinator only; P_L/P_G learn the local/graph PV only; J_L/J_G/J_NoMP learn both roles. Standalone L is the left-if-safe rule, not the local encoder. B1 is the canonical native-background reference, B2 reactive yielding, N externally commanded stay. D2_L is the fixed J_L coordinator with a native PV, labeled D_L in the manuscript.

There are five selected A and J_G models, three P_L/P_G/J_L and two J_NoMP. Original seeds are 20260914/15/16; additional seeds are 2026102804/05. The NoMP comparison uses only the two additional J_G models against the two NoMP models. Original and additional execution/training cohorts must not be treated as an architectural intervention.

Each episode key is `(method,model,d,f,road)`. `model=rule` denotes a fixed rule. The separate [road index](<data/road_keys.csv.gz>) identifies physical initial conditions; initial random streams do not imply identical realized later stochastic events after different actions. B1 has one physical series per density: fraction labels in plotted summaries are display aliases, not additional observations.

## Aggregation and intervals

Absolute means average available roads within each fixed model and then give those models equal weight. Original high-AV models have 30 roads (0–29); expanded panels have 42 (0–21 and 30–49). Their union is 50 and common intersection 22. Low-AV and matched two-seed encoder comparisons use 42 common roads. Paired family contrasts retain the intersection across every required fixed model, not a different subset for each subtraction. Missing technical records are not traffic failures or zero outcomes.

The archived speed intervals use 10,000 road-identity draws, seed 910019, linear 2.5/97.5 percentile endpoints and the supplied [original scalar bootstrap implementation](<analysis/global_stats.py>). Resampling is joint across the compared fixed models. Road uncertainty is conditional on those models, not uncertainty over retraining. These intervals are pointwise and not adjusted for a search over conditions. Counts of positive/negative intervals are descriptive.

An individual background-flow ratio is the mean of **paired roadwise ratios** to B1, with positive B1 denominators; it is not the ratio of two road means. Family summaries weight models equally. The 0.95 screen classifies observed point estimates and is not a per-road or deployment guarantee. The new individual flow intervals independently resample paired roads using `SeedSequence([910019,d,f,training_seed_or_zero,sum(ord(c) for c in method)])`; see [recomputation](<analysis/verify_flow_intervals.py>). No family-wide flow guarantee is inferred.

The fixed-coordinator diagnostic uses three J_L lineages and the same 22 roads in both arms at densities .20,.25,.30,.40,.55,.60 and f=100. Average `v` and stopped fraction `p` across these fixed models/roads, then define `u=v/(1-p)`. With L/N denoting learned/native PV:

```
vL-vN = (pN-pL)*(uL+uN)/2 + (uL-uN)*(1-(pL+pN)/2)
```

All ratios and both terms are recomputed inside each paired draw. The diagnostic uses `SeedSequence([910019,d,100,25])`. Its bootstrap stream differs from the archived speed-contrast stream, so small endpoint differences between the two valid calculations are expected. This is an algebraic attribution, not a measured frequency of command conflicts. [Diagnostic code](<analysis/jl_decomposition.py>) verifies its reference table by default; `--output NEWFILE` optionally writes a separate rebuilt table.

`completed_priority_laps` and completion probabilities include unsuccessful runs. `mean_priority_lap_time` is success-conditional; undefined values are not replaced with zero. Missing interval endpoints in the global table mean no interval is supplied for that estimand.

## Local profiles and training

The illustrated local condition is density .40, AV fraction .50. Windows exclude the PV and use post-transition observations, not the initial t=0 state. Primary moving width is 25 longitudinal cells across four lanes: area 100 sites. Density is background count/area; snapshot flow is background speed sum/area. This is a laboratory-frame window snapshot, not crossing-count or Edie flow.

For nonempty windows, instantaneous speed is speed sum/count and stopped fraction is zero-speed count/count. Averaging instantaneous ratios differs from pooling numerator/denominator. Empty windows have zero density/flow and undefined mean speed. Stationary windows and PV-moving windows have different membership and selection.

A restart is positive PV speed after at least five observed zero-speed steps. All-followup retains later stops; the moving-survivor cohort retains events uninterrupted through the current lag. Events average within roads before model aggregation. Event counts and frames are not independent Monte Carlo replicates. `L09` support columns describe contributing restarts per road; `roads_*` count roads with defined estimates. Figure 6 shading is the min–max of five model means, **not a confidence interval**.

The local tables retain aggregates/support, not raw event histories or per-step trajectories. Replotting and aggregation checks are supported; fresh extraction of local statistics is outside this package.

Training values are retained rolling training metrics, not held-out evaluation. There are 611 accepted iteration positions for each of 19 models (11,609 total), including missing metrics and restart segments. The two NoMP histories are unavailable. Selected models need not be final iteration 611; the [weights index](<weights/MODELS.json>) identifies exact selected checkpoints. Numeric policy weights do not contain optimizer/sampler/environment state and cannot support a claim of bitwise training continuation.
