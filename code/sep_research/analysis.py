"""Run-level descriptive and co-primary statistics; no simulator dependencies.

pointwise_summary consumes one condition and named treatment/comparator instances
plus fixed B1, after complete matrix validation. primary_analysis consumes all
five families, three training seeds and the fixed 12-condition support. Fixed A
appears once per training seed, shared by both contrasts. No test-based selection
or equivalence conclusion is implemented. Bootstrap coverage is approximate.
"""

import numpy as np

from snfs_traffic.metrics.priority import paired_bootstrap_interval
from .protocol import (
    BOOTSTRAP_ROOT, FAMILIES, FRACTION_TOKENS, METRICS, TRAINING_DENSITY_IDS,
    TRAINING_SEEDS, validate_pair, validate_rows,
)

CONTRASTS = (("J_L", "A"), ("J_L", "P_L"), ("J_G", "A"),
             ("J_G", "P_G"), ("J_G", "J_L"))


def _samples(samples):
    if type(samples) is not int or samples < 1:
        raise ValueError("samples must be an integer >= 1")


def _finite_vector(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError("expected a nonempty finite run-level vector")
    return values


def absolute_summary(values):
    """Equal run-weight mean, sample SEM and normal mean +/-1.96SEM.

    None denotes no completed lap; callers must use nulls only for lap time.
    Availability uses all runs. A single observed run has no estimable SEM/CI.
    """
    values = list(values)
    observed = [v for v in values if v is not None]
    n = len(observed)
    result = dict(runs=len(values), valid_runs=n,
                  availability=n / len(values) if values else None,
                  mean=None, sem=None, ci95=None, interval="normal_approximation_1.96SEM")
    if n:
        array = _finite_vector(observed)
        result["mean"] = float(array.mean())
        if n > 1:
            sem = float(array.std(ddof=1) / np.sqrt(n))
            result.update(sem=sem, ci95=[result["mean"] - 1.96 * sem, result["mean"] + 1.96 * sem])
    return result


def paired_summary(left, right, *, samples=10_000, seed=BOOTSTRAP_ROOT, allow_missing=False):
    """Pointwise 95% percentile CI for paired run differences, never step samples.

    allow_missing=True is for completed-lap times only. Keep a pair only when
    both values are observed, while exposing each side's completion availability.
    """
    _samples(samples)
    left, right = list(left), list(right)
    if len(left) != len(right) or not left:
        raise ValueError("paired vectors must have the same nonzero length")
    if not allow_missing and any(v is None for v in left + right):
        raise ValueError("missing non-lap metric")
    left_summary, right_summary = absolute_summary(left), absolute_summary(right)
    differences = [u - v for u, v in zip(left, right) if u is not None and v is not None]
    result = dict(runs=len(left), valid_pairs=len(differences), mean=None, ci95=None,
                  left_availability=left_summary["availability"],
                  right_availability=right_summary["availability"], reason=None)
    if differences:
        values = _finite_vector(differences)
        result.update(mean=float(values.mean()), ci95=list(paired_bootstrap_interval(
            values, samples=samples, seed=seed)))
    else:
        result["reason"] = "no_pairs_with_both_laps_observed"
    return result


def background_screen(method_flow, b1_flow, speed_difference, *, samples=10_000, seed=BOOTSTRAP_ROOT):
    """B1 runwise ratios and descriptive screen, with complete-ratio gates.

    speed_difference is a paired_summary for the explicitly named comparator.
    The flow denominator stays B1 even if that comparator is another policy.
    Undefined per-run ratios are null with a reason, never epsilon adjusted.
    """
    _samples(samples)
    numerator, denominator = _finite_vector(method_flow), _finite_vector(b1_flow)
    if numerator.shape != denominator.shape or np.any(numerator < 0) or np.any(denominator < 0):
        raise ValueError("flows must be paired and nonnegative")
    if speed_difference["runs"] != numerator.size:
        raise ValueError("speed and flow run counts differ")
    valid = denominator != 0
    ratios = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=valid)
    observed = ratios[valid]
    if not np.all(np.isfinite(observed)):
        raise ValueError("nonfinite flow ratio")
    complete = bool(np.all(valid))
    # Centering preserves a constant ratio exactly, including the .95 boundary.
    mean = float(observed[0] + np.mean(observed - observed[0])) if observed.size else None
    ci = list(paired_bootstrap_interval(observed, samples=samples, seed=seed)) if observed.size else None
    speed_ci = speed_difference["ci95"]
    passed = (complete and speed_difference["valid_pairs"] == numerator.size
              and numerator.size >= 3 and speed_ci is not None and speed_ci[0] > 0
              and mean is not None and mean >= .95)
    return dict(mean_of_runwise_flow_ratios=mean, ci95=ci, valid_ratio_pairs=int(valid.sum()),
                runs=int(numerator.size), complete=complete,
                ratios=[float(r) if ok else None for r, ok in zip(ratios, valid)],
                ratio_reasons=[None if ok else "zero_b1_background_flow" for ok in valid],
                reason=None if complete else "zero_b1_background_flow",
                background_difference=paired_summary(numerator, denominator, samples=samples, seed=seed),
                effective_region=bool(passed))


def pointwise_summary(rows, *, manifest, checkpoints, treatment, comparator,
                      samples=10_000, seed=BOOTSTRAP_ROOT):
    """Analyze one complete condition; instances are (method, training_seed).

    checkpoints must name exactly treatment, comparator and ('B1', None), with
    duplicates collapsed if B1 is itself the comparator. Return existing metric
    names for paired differences and absolute summaries plus a separate B1 screen.
    """
    validate_rows(rows, manifest=manifest, checkpoints=checkpoints)
    if len({(r.canonical_density_id, r.fraction_token) for r in manifest}) != 1:
        raise ValueError("pointwise analysis requires one condition")
    if set(checkpoints) != {treatment, comparator, ("B1", None)}:
        raise ValueError("pointwise checkpoint map must name treatment/comparator/B1")
    if treatment == comparator or treatment == ("B1", None):
        raise ValueError("treatment must differ from comparator and B1")
    if (treatment[1] is not None and comparator[1] is not None
            and treatment[1] != comparator[1]):
        raise ValueError("pointwise comparison must match training seeds")
    # Only join after duplicate, completeness, and provenance validation.
    indexed = {(r["method"], r["training_seed"], r["run_index"]): r for r in rows}
    runs = sorted(r.run_index for r in manifest)
    left = [indexed[(*treatment, run)] for run in runs]
    right = [indexed[(*comparator, run)] for run in runs]
    baseline = [indexed[("B1", None, run)] for run in runs]
    for u, v in zip(left, right):
        validate_pair(u, v)
    differences, absolute = {}, {}
    for i, metric in enumerate(METRICS):
        u, v = [r[metric] for r in left], [r[metric] for r in right]
        differences[metric] = paired_summary(u, v, samples=samples, seed=seed + i,
                                             allow_missing=metric == "mean_priority_lap_time")
        absolute[metric] = dict(treatment=absolute_summary(u), comparator=absolute_summary(v))
    screen = background_screen([r["flow_background"] for r in left],
                               [r["flow_background"] for r in baseline],
                               differences["mean_speed_priority"], samples=samples, seed=seed + len(METRICS))
    return dict(treatment=list(treatment), comparator=list(comparator), differences=differences,
                absolute=absolute, background_screen=screen, bootstrap_seed=seed,
                bootstrap_samples=samples, interpretation="pointwise_descriptive_not_equivalence")


def crossed_bootstrap_means(values, *, samples=10_000, seed=BOOTSTRAP_ROOT):
    """Return [resample,family] means for [family,seed,density,fraction,run].

    Low-level kernel permits small arrays for analytic tests; primary_analysis
    enforces the preregistered (5,3,4,3,30) matrix. For each resample draw seed
    indices once, then independently draw run indices in each fixed density.
    Share both draws across families, and run draws across fractions and seeds.
    Densities/fractions retain equal weights and are never resampled.
    """
    _samples(samples)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 5 or any(n < 1 for n in values.shape) or not np.all(np.isfinite(values)):
        raise ValueError("expected finite nonempty [family,seed,density,fraction,run] array")
    families, seeds, densities, fractions, runs = values.shape
    rng = np.random.default_rng(seed)
    result = np.empty((samples, families), dtype=np.float64)
    for b in range(samples):
        seed_indices = rng.integers(seeds, size=seeds)
        total = np.zeros(families)
        for density in range(densities):
            run_indices = rng.integers(runs, size=runs)
            block = np.take(values[:, :, density, :, :], seed_indices, axis=1)
            block = np.take(block, run_indices, axis=3)
            total += block.mean(axis=(1, 2, 3))
        result[b] = total / densities
    return result


def primary_from_array(values, *, samples=10_000, seed=BOOTSTRAP_ROOT):
    """Analyze complete speed array (5,3,4,3,30), ordered by protocol constants.

    Numerical entry point assumes provenance already checked; evaluators should
    call primary_analysis. A has exactly one family axis entry, never duplicated.
    All five contrasts use identical bootstrap draws. CI99 endpoints .005/.995.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (5, 3, 4, 3, 30) or not np.all(np.isfinite(values)):
        raise ValueError("primary matrix must be complete finite (5,3,4,3,30)")
    boot = crossed_bootstrap_means(values, samples=samples, seed=seed)
    estimates = values.mean(axis=(2, 3, 4))
    results = {}
    for left, right in CONTRASTS:
        i, j = FAMILIES.index(left), FAMILIES.index(right)
        per_seed = estimates[i] - estimates[j]
        distribution = boot[:, i] - boot[:, j]
        ci95 = np.quantile(distribution, [.025, .975]).tolist()
        ci99 = np.quantile(distribution, [.005, .995]).tolist()
        results[f"{left}-{right}"] = dict(mean=float(per_seed.mean()),
            per_training_seed={str(k): float(v) for k, v in zip(TRAINING_SEEDS, per_seed)},
            ci95=ci95, ci99_adjusted=ci99, positive_adjusted=bool(ci99[0] > 0))
    return dict(contrasts=results, bootstrap_seed=seed, bootstrap_samples=samples,
                ci95_endpoints=[.025, .975], ci99_adjusted_endpoints=[.005, .995],
                resampling="crossed_training_seed_and_density_stratified_run_blocks",
                condition_weight=1 / 12, density_ids=list(TRAINING_DENSITY_IDS),
                fraction_tokens=list(FRACTION_TOKENS), training_seeds=list(TRAINING_SEEDS),
                coverage="approximate; three training seeds; zero-crossing is not equivalence")


def primary_analysis(rows, *, manifest, checkpoints, samples=10_000, seed=BOOTSTRAP_ROOT):
    """Validate provenance/completeness, then analyze the fixed training support.

    Accept exactly the 15 learned instances and their 12*30 run rows. Select this
    preregistered subset from a separately validated full-test matrix; fixed
    controls do not enter the co-primary bootstrap.
    """
    validate_rows(rows, manifest=manifest, checkpoints=checkpoints)
    if set(checkpoints) != {(f, s) for f in FAMILIES for s in TRAINING_SEEDS}:
        raise ValueError("primary requires five families and three matched training seeds")
    conditions = {(r.canonical_density_id, r.fraction_token, r.run_index) for r in manifest}
    expected = {(d, f, r) for d in TRAINING_DENSITY_IDS for f in FRACTION_TOKENS for r in range(30)}
    if conditions != expected:
        raise ValueError("primary requires fixed 12 training-support conditions and 30 runs")
    if {r.split for r in manifest} != {"test"}:
        raise ValueError("primary requires final test split")
    indexed = {(r["method"], r["training_seed"], r["canonical_density_id"],
                r["fraction_token"], r["run_index"]): r["mean_speed_priority"] for r in rows}
    values = np.array([[[[[indexed[(m, s, d, f, r)] for r in range(30)]
                         for f in FRACTION_TOKENS] for d in TRAINING_DENSITY_IDS]
                       for s in TRAINING_SEEDS] for m in FAMILIES])
    return primary_from_array(values, samples=samples, seed=seed)
