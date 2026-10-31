"""Pure fixed-model MC statistics; column identity must be proved upstream."""
from functools import lru_cache
import math

SAMPLES = 10000
BOOTSTRAP_ROOT = 910019
INTERVAL_KIND = 'pointwise_conditional_mc_bootstrap_fixed_models'


def _numpy():
    import numpy as np
    return np


def _number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _matrix(values):
    np = _numpy()
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError('expected model x repeat matrix with at least one model')
    if np.isinf(values).any():
        raise ValueError('infinite values are invalid')
    return values


def _model_summary(values):
    np = _numpy()
    present = np.isfinite(values)
    counts = present.sum(axis=1)
    means = [(_number(row[mask].mean()) if n else None)
             for row, mask, n in zip(values, present, counts)]
    complete = all(value is not None for value in means)
    return dict(model_counts=counts.tolist(), model_means=means,
                mean=_number(np.mean(means)) if complete else None,
                training_sample_sd=(_number(np.std(means, ddof=1))
                                    if complete and len(means) >= 2 else None))


@lru_cache(maxsize=8)
def _counts(n):
    """Owned RNG and bounded cache; never allocate a draws x repeat x repeat cube."""
    np = _numpy()
    draws = np.random.default_rng(BOOTSTRAP_ROOT).integers(0, n, (SAMPLES, n))
    counts = np.zeros((SAMPLES, n), dtype=np.int64)
    np.add.at(counts, (np.arange(SAMPLES)[:, None], draws), 1)
    counts.flags.writeable = False
    return counts


def _interval(samples, method, reason=None):
    np = _numpy()
    valid = np.isfinite(samples)
    n_valid = int(valid.sum())
    if reason is None and n_valid != SAMPLES:
        reason = 'invalid_bootstrap_draws'
    bounds = None
    if reason is None:
        bounds = [_number(v) for v in np.percentile(
            samples, [2.5, 97.5], method='linear')]
        if any(v is None for v in bounds):
            bounds = None
            reason = 'nonfinite_bootstrap_interval'
    return dict(ci95=bounds, ci_reason=reason, n_boot_valid=n_valid,
                bootstrap_method=method, interval_kind=INTERVAL_KIND)


def _no_interval(method, reason):
    # Zero means no valid draws evaluated; see ci_reason for the precondition.
    return dict(ci95=None, ci_reason=reason, n_boot_valid=0,
                bootstrap_method=method, interval_kind=INTERVAL_KIND)


def absolute(values):
    """Equal-model mean of available MC means, with casewise union bootstrap."""
    np = _numpy()
    values = _matrix(values)
    present = np.isfinite(values)
    n_union = values.shape[1]
    if n_union == 0 or (~present.any(axis=0)).any():
        raise ValueError('union must be nonempty and contain no all-missing column')
    result = dict(_model_summary(values), n_union=n_union, reason=None)
    method = 'casewise_union_equal_model'
    if min(result['model_counts']) == 0:
        result['reason'] = 'required_model_missing'
    elif result['mean'] is None:
        result['reason'] = 'nonfinite_aggregate'
    if result['reason'] is not None:
        result.update(_no_interval(method, result['reason']))
    elif min(result['model_counts']) < 2:
        result.update(_no_interval(method, 'fewer_than_two_per_model'))
    else:
        counts = _counts(n_union)
        denominators = counts @ present.T.astype(float)
        # Zero is only the additive identity for masked sums, never an imputed episode.
        totals = counts @ np.where(present, values, 0.0).T
        model_means = np.full(totals.shape, np.nan)
        np.divide(totals, denominators, out=model_means, where=denominators > 0)
        samples = model_means.mean(axis=1)
        reason = ('bootstrap_missing_required_model'
                  if (denominators == 0).any() else None)
        result.update(_interval(samples, method, reason))
    return result


def paired_contrast(left, right):
    """Complete-intersection contrast; a one-row shared baseline is subtracted once."""
    np = _numpy()
    left, right = _matrix(left), _matrix(right)
    if left.shape[1] != right.shape[1]:
        raise ValueError('aligned sides must have equal repeat-axis lengths')
    selected = np.flatnonzero(np.isfinite(left).all(axis=0)
                              & np.isfinite(right).all(axis=0))
    n = len(selected)
    values = left[:, selected].mean(axis=0) - right[:, selected].mean(axis=0)
    mean = _number(values.mean()) if n else None
    result = dict(n_pairs=n, selected_column_indices=selected.tolist(), mean=mean,
                  left_model_counts=np.isfinite(left).sum(axis=1).tolist(),
                  right_model_counts=np.isfinite(right).sum(axis=1).tolist(),
                  reason='no_complete_pairs' if not n else None, sign=None)
    method = 'complete_intersection_paired'
    if n < 2:
        result.update(_no_interval(method, 'fewer_than_two_pairs'))
    else:
        samples = (_counts(n) @ values) / n
        result.update(_interval(samples, method))
        if result['ci95'] is not None:
            low, high = result['ci95']
            result['sign'] = 1 if low > 1e-12 else -1 if high < -1e-12 else 0
    if n and mean is None:
        result['reason'] = 'nonfinite_aggregate'
    return result


def background_ratio(a, b):
    """Mean of paired lineage/B1 ratios; unknown screen for undefined estimates."""
    np = _numpy()
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError('expected aligned one-dimensional arrays')
    if np.isinf(a).any() or np.isinf(b).any() or (a < 0).any() or (b < 0).any():
        raise ValueError('metrics must be nonnegative finite values or technical NaN')
    selected = np.flatnonzero(np.isfinite(a) & np.isfinite(b))
    n = len(selected)
    valid = int((b[selected] > 0).sum())
    reason = 'no_matched_pairs' if not n else 'nonpositive_denominator' if valid != n else None
    mean = _number(np.mean(a[selected] / b[selected])) if reason is None else None
    if reason is None and mean is None:
        reason = 'nonfinite_aggregate'
    return dict(mean=mean, reason=reason, n_pairs=n,
                selected_column_indices=selected.tolist(),
                required_denominators=n, valid_denominators=valid,
                screen_ge_095=bool(mean >= .95) if mean is not None else None)


def conditional_laps(values, observed):
    """Separate technical absence, observed no-lap episodes, and positive lap means."""
    np = _numpy()
    values = _matrix(values)
    observed = np.asarray(observed)
    if observed.dtype != np.dtype(bool) or observed.shape != values.shape:
        raise ValueError('observed must be a boolean mask matching values')
    completed = np.isfinite(values)
    if (completed & ~observed).any() or (values <= 0).any():
        raise ValueError('finite lap means must be positive and observed')
    completion = absolute(np.where(observed, completed.astype(float), np.nan))
    laps = _model_summary(values)
    observed_counts = observed.sum(axis=1)
    reason = ('required_model_no_observations' if (observed_counts == 0).any()
              else 'required_model_no_completed_laps' if min(laps['model_counts']) == 0
              else 'nonfinite_aggregate' if laps['mean'] is None else None)
    return dict(model_successes=laps['model_counts'],
                model_observed_counts=observed_counts.tolist(),
                model_completion_rates=completion['model_means'],
                model_lap_means=laps['model_means'], mean=laps['mean'],
                training_sample_sd=laps['training_sample_sd'], reason=reason,
                ci95=None, ci_reason='conditional_lap_mean_no_interval',
                completion=completion)
