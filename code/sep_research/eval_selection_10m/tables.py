"""Aggregated dense-grid tables for the 12-pair 10M held-out test matrix.

Reads the verified `raw.json` rows (24,840 measurement trajectories over
18 densities x 3 AV fractions x 30 runs, plus B1/B2/N/L controls) and writes
machine-readable condition tables plus a short human-readable summary.

Reuses `sep_research.analysis.absolute_summary` / `paired_summary` for the
pointwise normal and paired-bootstrap intervals, exactly as the previous
study's `preliminary_gpu_paper/figures.py` did. Differences that required
adaptation are recorded in `ADAPTATION_NOTES` and in the emitted metadata.
"""
import csv
import json
from itertools import product
from pathlib import Path

import numpy as np

from sep_research import protocol
from sep_research.analysis import absolute_summary, paired_summary
from sep_research.eval_selection_10m import selection as sel

FAMILIES = sel.FAMILIES
CONTROLS = ('B1', 'B2', 'N', 'L')
COMPARATORS = ('B2', 'B1', 'N', 'L')
PRIMARY = 'mean_speed_priority'
METRICS = ('mean_speed_priority', 'flow_all', 'flow_background',
           'stopped_fraction_priority', 'mean_priority_lap_time')
LAP = 'mean_priority_lap_time'
DENSITIES = protocol.DENSITY_TOKENS
FRACTIONS = protocol.FRACTION_TOKENS
RUNS = 30
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = protocol.BOOTSTRAP_ROOT
ADAPTATION_NOTES = (
    'vs preliminary_gpu_paper/figures.py: dense 18-density grid (5..90 step 5) instead of 8 '
    'scouting densities; 30 runs instead of 5; three training seeds instead of one, so absolute '
    'and paired intervals cover 90 (family) or 30 (control) run-level observations and per-seed '
    'means are reported separately; four families A/P_L/P_G/J_L (J_G deferred); the selected '
    'checkpoint is per (family, seed) from sep_selection_10m_v1, not per family; flow ratios and '
    'lap-time availability keep the earlier descriptive rules.')


def load_rows(path):
    value = json.loads(Path(path).read_text())
    rows = value['rows'] if isinstance(value, dict) else value
    if not isinstance(rows, list) or not rows:
        raise ValueError('empty raw rows')
    return rows


def index_rows(rows, *, families=FAMILIES, seeds=protocol.TRAINING_SEEDS,
               control_modes=CONTROLS, runs=RUNS):
    learned, controls = {}, {}
    for row in rows:
        density, fraction, run = row['density_token'], row['fraction_token'], row['run_index']
        if row['method'] in families:
            key = (row['method'], row['training_seed'], density, fraction, run)
            if key in learned:
                raise ValueError(f'duplicate learned row {key}')
            learned[key] = row
        elif row['method'] in CONTROLS:
            key = (row['method'], density, fraction, run)
            if key in controls:
                raise ValueError(f'duplicate control row {key}')
            controls[key] = row
        else:
            raise ValueError(f'unknown method {row["method"]}')
    for family in families:
        for density, fraction in product(DENSITIES, FRACTIONS):
            for seed in seeds:
                for run in range(runs):
                    if (family, seed, density, fraction, run) not in learned:
                        raise ValueError(f'missing learned row {(family, seed, density, fraction, run)}')
    for control in control_modes:
        for density, fraction in product(DENSITIES, FRACTIONS):
            for run in range(runs):
                if (control, density, fraction, run) not in controls:
                    raise ValueError(f'missing control row {(control, density, fraction, run)}')
    return learned, controls


def _values(rows, metric, *, allow_missing=False):
    values = [row[metric] for row in rows]
    if allow_missing:
        return values
    if any(value is None for value in values):
        raise ValueError(f'missing {metric}')
    return values


def condition_records(rows, *, families=FAMILIES, seeds=protocol.TRAINING_SEEDS,
                      control_modes=CONTROLS, metrics=METRICS, runs=RUNS):
    learned, controls = index_rows(rows, families=families, seeds=seeds,
                                   control_modes=control_modes, runs=runs)
    records = []
    for family, density, fraction in product(families, DENSITIES, FRACTIONS):
        by_seed = {seed: [learned[(family, seed, density, fraction, run)] for run in range(runs)]
                   for seed in seeds}
        pooled = [row for seed in seeds for row in by_seed[seed]]
        for metric in metrics:
            allow = metric == LAP
            summary = absolute_summary(_values(pooled, metric, allow_missing=allow))
            seed_means = {str(seed): absolute_summary(_values(by_seed[seed], metric,
                                                             allow_missing=allow))['mean']
                          for seed in seeds}
            records.append(dict(kind='absolute', family=family, method=family, comparator='',
                density_token=density, fraction_token=fraction, metric=metric, n_total=len(pooled),
                n=summary['valid_runs'], mean=summary['mean'], sem=summary['sem'],
                ci_low=None if summary['ci95'] is None else summary['ci95'][0],
                ci_high=None if summary['ci95'] is None else summary['ci95'][1],
                seed_means=seed_means,
                seed_mean_of_means=float(np.mean([v for v in seed_means.values() if v is not None]))
                if any(v is not None for v in seed_means.values()) else None,
                seed_sem=(float(np.std([v for v in seed_means.values() if v is not None], ddof=1)
                                / np.sqrt(len(seeds)))
                          if sum(v is not None for v in seed_means.values()) > 1 else None)))
        for comparator in COMPARATORS:
            control = [controls[(comparator, density, fraction, run)] for run in range(runs)]
            for metric in metrics:
                allow = metric == LAP
                left = _values(pooled, metric, allow_missing=allow)
                right = _values(control, metric, allow_missing=allow) * len(seeds)
                if allow and all(v is None for v in left + right):
                    continue
                summary = paired_summary(left, right, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED,
                                         allow_missing=allow)
                records.append(dict(kind='paired', family=family, method=family, comparator=comparator,
                    density_token=density, fraction_token=fraction, metric=metric,
                    n_total=len(left), n=summary['valid_pairs'], mean=summary['mean'],
                    sem=None,
                    ci_low=None if summary['ci95'] is None else summary['ci95'][0],
                    ci_high=None if summary['ci95'] is None else summary['ci95'][1],
                    seed_means={}, seed_mean_of_means=None, seed_sem=None))
        ratios = [a['flow_background'] / b['flow_background']
                  for a, b in zip(pooled, control * len(seeds))
                  if b['flow_background'] != 0]
        point = absolute_summary(ratios) if ratios else dict(valid_runs=0, mean=None, ci95=None)
        records.append(dict(kind='ratio', family=family, method=family, comparator='B1',
            density_token=density, fraction_token=fraction, metric='background_ratio',
            n_total=len(pooled), n=point['valid_runs'], mean=point['mean'], sem=None,
            ci_low=None if point['ci95'] is None else point['ci95'][0],
            ci_high=None if point['ci95'] is None else point['ci95'][1],
            screen_ge_095=(point['mean'] >= .95) if point['mean'] is not None else None,
            seed_means={}, seed_mean_of_means=None, seed_sem=None))
    for control in control_modes:
        for density, fraction in product(DENSITIES, FRACTIONS):
            group = [controls[(control, density, fraction, run)] for run in range(runs)]
            for metric in metrics:
                summary = absolute_summary(_values(group, metric, allow_missing=metric == LAP))
                records.append(dict(kind='control', family='', method=control, comparator='',
                    density_token=density, fraction_token=fraction, metric=metric,
                    n_total=len(group), n=summary['valid_runs'], mean=summary['mean'],
                    sem=summary['sem'],
                    ci_low=None if summary['ci95'] is None else summary['ci95'][0],
                    ci_high=None if summary['ci95'] is None else summary['ci95'][1],
                    seed_means={}, seed_mean_of_means=None, seed_sem=None))
    return records


def family_summary(records, *, families=FAMILIES, metrics=METRICS):
    """Equal-weight aggregation over conditions; between-condition SEM (descriptive)."""
    summary = []
    for family in families:
        for metric in metrics:
            for scope, fractions in [('overall', FRACTIONS)] + [(f'AV{value}', (value,)) for value in FRACTIONS]:
                means = [r['mean'] for r in records if r['kind'] == 'absolute' and r['family'] == family
                         and r['metric'] == metric and r['fraction_token'] in fractions
                         and r['mean'] is not None]
                if not means:
                    continue
                summary.append(dict(family=family, metric=metric, scope=scope, conditions=len(means),
                    condition_mean=float(np.mean(means)),
                    condition_sem=float(np.std(means, ddof=1) / np.sqrt(len(means))) if len(means) > 1 else None,
                    condition_min=float(np.min(means)), condition_max=float(np.max(means))))
            for comparator in COMPARATORS:
                deltas = [r['mean'] for r in records if r['kind'] == 'paired' and r['family'] == family
                          and r['comparator'] == comparator and r['metric'] == metric
                          and r['mean'] is not None]
                if not deltas:
                    continue
                summary.append(dict(family=family, metric=metric, scope=f'vs_{comparator}',
                    conditions=len(deltas), condition_mean=float(np.mean(deltas)),
                    condition_sem=float(np.std(deltas, ddof=1) / np.sqrt(len(deltas))) if len(deltas) > 1 else None,
                    condition_min=float(np.min(deltas)), condition_max=float(np.max(deltas))))
    return summary


def _write_csv(path, records, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: (json.dumps(record.get(key), sort_keys=True)
                                   if isinstance(record.get(key), (dict, list)) else record.get(key))
                             for key in fields})


CONDITION_FIELDS = ('kind', 'family', 'method', 'comparator', 'density_token', 'fraction_token',
                    'metric', 'n_total', 'n', 'mean', 'sem', 'ci_low', 'ci_high', 'screen_ge_095',
                    'seed_means', 'seed_mean_of_means', 'seed_sem')
SUMMARY_FIELDS = ('family', 'metric', 'scope', 'conditions', 'condition_mean', 'condition_sem',
                  'condition_min', 'condition_max')


def build_table_set(raw_path, out_dir, *, selection_path=None, plan_path=None, counts_path=None):
    rows = load_rows(raw_path)
    records = condition_records(rows)
    summary = family_summary(records)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(schema='sep_tables_10m_v1', variant='12_pair_jg_deferred',
        raw=str(raw_path), raw_rows=len(rows),
        densities=list(DENSITIES), fractions=list(FRACTIONS), runs=RUNS,
        training_seeds=list(protocol.TRAINING_SEEDS), families=list(FAMILIES),
        controls=list(CONTROLS), metrics=list(METRICS),
        bootstrap_samples=BOOTSTRAP_SAMPLES, bootstrap_seed=BOOTSTRAP_SEED,
        interval_scope=('pointwise run-level normal CI for absolute means; paired percentile '
                        'bootstrap CI over 30 runs x 3 seeds; seed means reported separately'),
        adaptation_notes=ADAPTATION_NOTES, heldout_test_ready=False,
        selection_sha256=None, plan_sha256=None, counts=None)
    for key, path in (('selection_sha256', selection_path), ('plan_sha256', plan_path)):
        if path is not None:
            import hashlib
            meta[key] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if counts_path is not None:
        meta['counts'] = json.loads(Path(counts_path).read_text())
    (out / 'conditions.json').write_text(json.dumps(dict(meta, records=records), sort_keys=True, indent=1) + '\n')
    (out / 'family_summary.json').write_text(json.dumps(dict(meta, summary=summary), sort_keys=True, indent=1) + '\n')
    _write_csv(out / 'conditions.csv', records, CONDITION_FIELDS)
    _write_csv(out / 'family_summary.csv', summary, SUMMARY_FIELDS)
    (out / 'SUMMARY.md').write_text(render_summary(meta, records, summary))
    return meta, records, summary


def _fmt(value, digits=4):
    return '—' if value is None else f'{value:.{digits}f}'


def render_summary(meta, records, summary):
    lines = ['# 10M CPU matrix — held-out test aggregates (12 pairs, J_G deferred)', '',
             f"Rows: {meta['raw_rows']}; grid: {len(meta['densities'])} densities x {len(meta['fractions'])} "
             f"AV fractions x {meta['runs']} runs x {len(meta['training_seeds'])} training seeds; "
             f"controls {'/'.join(meta['controls'])}.", '',
             'Absolute values are equal-weight means over run-level observations (90 per condition for '
             'learned families, 30 for controls); paired advantages use the paired percentile bootstrap '
             'over the same 30 runs x 3 seeds. Between-seed means are listed separately in '
             '`conditions.csv` (`seed_means`).', '',
             '## Primary metric: mean PV speed by family (overall and per AV fraction)', '',
             '| family | scope | conditions | mean | SEM across conditions | min | max |', '|---|---|---:|---:|---:|---:|---:|']
    for row in summary:
        if row['metric'] != PRIMARY or row['scope'].startswith('vs_'):
            continue
        lines.append(f"| {row['family']} | {row['scope']} | {row['conditions']} | {_fmt(row['condition_mean'])} | "
                     f"{_fmt(row['condition_sem'])} | {_fmt(row['condition_min'])} | {_fmt(row['condition_max'])} |")
    lines += ['', '## Mean paired advantage in PV speed (equal weight over the 54 conditions)', '',
              '| family | comparator | conditions | mean Δ | SEM across conditions | min | max |', '|---|---|---:|---:|---:|---:|---:|']
    for row in summary:
        if row['metric'] != PRIMARY or not row['scope'].startswith('vs_'):
            continue
        lines.append(f"| {row['family']} | {row['scope'][3:]} | {row['conditions']} | {_fmt(row['condition_mean'])} | "
                     f"{_fmt(row['condition_sem'])} | {_fmt(row['condition_min'])} | {_fmt(row['condition_max'])} |")
    control_means = {}
    for control in CONTROLS:
        values = [r['mean'] for r in records if r['kind'] == 'control' and r['method'] == control
                  and r['metric'] == PRIMARY and r['mean'] is not None]
        control_means[control] = float(np.mean(values)) if values else None
    lines += ['', '## Control reference (mean PV speed, equal weight over the 54 conditions)', '',
              '| control | mean PV speed |', '|---|---:|']
    for control, value in control_means.items():
        lines.append(f'| {control} | {_fmt(value)} |')
    lines += ['', '## Caveats', '',
              '* 12-pair matrix only; J_G is deferred to a separate run and command.',
              '* Selected checkpoint per (family, training seed) comes from the validation-stage '
              '`sep_selection_10m_v1` receipt; 11 of 12 pairs were within the <0.05 tie threshold and '
              'were resolved by the 50-run fresh-seed tie-break with an earlier-checkpoint fallback.',
              '* Intervals are pointwise and descriptive; no multiplicity correction and no formal '
              'non-inferiority claim.',
              '* The frozen 20M held-out gate (15 anchored pairs) is untouched; this is the labelled '
              'intermediate 12-pair variant path.', '']
    return '\n'.join(lines)


def plot_figures(records, rows, out_dir, *, families=FAMILIES):
    """Journal-facing rendering of the 10M dense-grid condition estimates.

    Typography and panel layout follow the CRM submission of the preliminary
    series (`scripts/typeset_figures.py`): full-text-width panels, one dash
    pattern per method so curves stay distinguishable in monochrome print,
    units on every axis, no shaded panel background, a single legend below the
    panels and vector output. Values are the same condition estimates written
    to conditions.json, never re-derived.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from paper import plot_style

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'axes.linewidth': .65, 'mathtext.default': 'regular',
                         'savefig.dpi': 300})
    control_style = dict(zip(CONTROLS, ('baseline1_hdv', 'baseline2_matched', 'noop', 'left_if_safe')))
    family_style = {'A': ('#0072B2', 's', '-'),
                    'P_L': ('#56B4E9', '*', (0, (6, 1, 1, 1))),
                    'P_G': ('#D55E00', '^', (0, (7, 2))),
                    'J_L': ('#009E73', 'P', (0, (2, 1))),
                    'J_G': ('#CC79A7', 'D', (0, (4, 1, 1, 1, 1, 1)))}
    labels = {'A': 'A', 'P_L': r'$P_{\mathrm{L}}$', 'P_G': r'$P_{\mathrm{G}}$',
              'J_L': r'$J_{\mathrm{L}}$', 'J_G': r'$J_{\mathrm{G}}$',
              'B1': 'B1', 'B2': 'B2', 'N': 'N', 'L': 'L'}
    lookup = {(r['kind'], r['metric'], r['method'], r['comparator'], r['density_token'], r['fraction_token']): r
              for r in records}

    def style(method):
        if method in control_style:
            key = control_style[method]
            return dict(color=plot_style.COLORS[key], marker=plot_style.MARKERS[key],
                        linestyle=plot_style.LINESTYLES[key],
                        markerfacecolor=plot_style.MARKER_FACE_COLORS[key])
        color, marker, linestyle = family_style[method]
        return dict(color=color, marker=marker, linestyle=linestyle)

    specs = {
        'pv_speed': [('absolute', PRIMARY, '', 'mean', 'Скорость PV,\nклеток/шаг')],
        'paired_deltas': [('paired', PRIMARY, b, 'mean', f'Δ скорости к {labels[b]},\nклеток/шаг')
                          for b in COMPARATORS],
        'flows': [('absolute', 'flow_all', '', 'mean', 'Общий поток,\nавт./(шаг·полосу)'),
                  ('absolute', 'flow_background', '', 'mean', 'Фоновый поток,\nавт./(шаг·полосу)')],
        'stopped_fraction': [('absolute', 'stopped_fraction_priority', '', 'mean',
                              'Доля времени\nбез движения')],
    }
    for name, panels in specs.items():
        nrows = len(panels)
        figure, axes = plt.subplots(nrows, 3, figsize=(5.65, 1.70 * nrows + 1.10),
                                    squeeze=False, sharex=True, sharey='row')
        for i, (kind, metric, comparator, value_key, ylabel) in enumerate(panels):
            for j, fraction in enumerate(FRACTIONS):
                ax = axes[i, j]
                ax.axvline(.40, color='black', alpha=.50, linewidth=.55, linestyle=':', zorder=0)
                for method in list(families) + list(CONTROLS):
                    row_kind = 'control' if method in CONTROLS else kind
                    series = [lookup.get((row_kind, metric, method, comparator, density, fraction))
                              for density in DENSITIES]
                    if any(item is None for item in series):
                        continue
                    x = np.asarray(DENSITIES) / 100
                    y = [item.get(value_key) if item.get(value_key) is not None else np.nan for item in series]
                    line, = ax.plot(x, y, label=labels[method], markersize=3.3, linewidth=1.05,
                                    markeredgewidth=.7, **style(method))
                    if value_key == 'mean':
                        ax.fill_between(x, [item['ci_low'] if item['ci_low'] is not None else np.nan for item in series],
                                        [item['ci_high'] if item['ci_high'] is not None else np.nan for item in series],
                                        color=line.get_color(), alpha=.09, linewidth=0)
                if kind == 'paired' and value_key == 'mean':
                    ax.axhline(0, color='black', linewidth=.6)
                ax.set_xlim(.04, .91)
                ax.set_xticks((.1, .3, .5, .7, .9))
                if i == 0:
                    ax.set_title(f'Доля AV = {fraction / 100:.2f}', fontsize=9, pad=6)
                if j == 0:
                    ax.set_ylabel(ylabel, labelpad=5)
                if i == nrows - 1:
                    ax.set_xlabel('Плотность ρ', labelpad=3)
                ax.spines[['top', 'right']].set_visible(False)
                ax.grid(alpha=.18, linewidth=.45)
                ax.tick_params(length=2.8, width=.6, pad=2)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        found = {}
        for ax in axes.ravel():
            handles, handle_labels = ax.get_legend_handles_labels()
            for handle, text in zip(handles, handle_labels):
                found.setdefault(text, handle)
        if found:
            figure.legend(list(found.values()), list(found), loc='lower center',
                          bbox_to_anchor=(.52, .01), ncol=4, frameon=False,
                          columnspacing=1.3, handlelength=2.3, handletextpad=.5, fontsize=9)
        figure.subplots_adjust(left=.16, right=.987, top=.91 if nrows == 1 else .95,
                               bottom=1.08 / (1.70 * nrows + 1.10), wspace=.13, hspace=.20)
        for suffix in ('pdf', 'svg'):
            figure.savefig(out / f'{name}.{suffix}', bbox_inches='tight', pad_inches=0.045)
        figure.savefig(out / f'{name}.png', bbox_inches='tight', pad_inches=0.045)
        plt.close(figure)
    return sorted(str(path) for path in out.glob('*'))
