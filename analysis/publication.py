"""Offline main-series estimates from immutable saved measurements; no evaluator imports."""
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np

REVIEW = Path(__file__).resolve().parents[1]
FAMILIES = ('A', 'P_L', 'P_G', 'J_L', 'J_G')
CONTROLS = ('B1', 'B2', 'N', 'L')
METHODS = FAMILIES + CONTROLS
SEEDS = (20260914, 20260915, 20260916)
DENSITIES = tuple(range(5, 91, 5))
FRACTIONS = (90, 95, 100)
METRICS = ('mean_speed_priority', 'flow_all', 'flow_background', 'stopped_fraction_priority')
LAP = 'mean_priority_lap_time'
BOOTSTRAP_ROOT = 910019  # sep_research/protocol.py; literal avoids scientific imports.
SAMPLES = 10000
PAIRS = (('J_L', 'A'), ('J_G', 'A'), ('J_L', 'P_L'), ('J_G', 'P_G'),
         ('P_G', 'P_L'), ('J_G', 'J_L'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n')


def indices(n=30):
    return np.random.default_rng(BOOTSTRAP_ROOT).integers(0, n, (SAMPLES, n))


def interval(values, draws):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None
    return np.percentile(values[draws].mean(axis=1), [2.5, 97.5], method='linear').tolist()


def interval_sign(bounds):
    return 1 if bounds[0] > 1e-12 else -1 if bounds[1] < -1e-12 else 0


def seed_stats(values):
    values = list(values)
    complete = all(v is not None for v in values)
    return dict(seed_means=dict(zip(map(str, SEEDS), values)),
                mean=float(np.mean(values)) if complete else None,
                seed_sd=float(np.std(values, ddof=1)) if complete else None)


def road_difference(left, right):
    """Each input has model x road shape; a shared control is subtracted once."""
    return np.asarray(left).mean(axis=0) - np.asarray(right).mean(axis=0)


def background_line(values, b1):
    a, b = np.asarray(values, dtype=float), np.asarray(b1, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b) & (b > 0)
    n = int(valid.sum())
    mean = float(np.mean(a / b)) if n == len(b) else None
    return dict(valid_denominators=n, required_denominators=len(b), mean=mean,
                screen_ge_095=bool(mean >= .95) if mean is not None else False)


def lap_line(values, right=None):
    valid = [i for i, v in enumerate(values) if v is not None and
             (right is None or right[i] is not None)]
    a = [values[i] if right is None else values[i] - right[i] for i in valid]
    return dict(successful_run_ids=valid, n=len(a), missing=len(values)-len(a),
                mean=float(np.mean(a)) if a else None,
                ci95=interval(a, indices(len(a))) if right is not None and len(a) >= 2 else None)


def index_rows(rows, families=FAMILIES, seeds=SEEDS, densities=DENSITIES, fractions=FRACTIONS, runs=30):
    indexed = {}
    for r in rows:
        key = (r['method'], r['training_seed'], r['density_token'], r['fraction_token'], r['run_index'])
        if key in indexed:
            raise ValueError(f'duplicate row: {key}')
        indexed[key] = r
    expected = {(m, s, d, f, run) for m in families + CONTROLS
                for s in (seeds if m in families else (None,))
                for d, f, run in product(densities, fractions, range(runs))}
    if set(indexed) != expected:
        raise ValueError(f'row coverage: missing {len(expected-set(indexed))}, extra {len(set(indexed)-expected)}')
    return indexed


def calculate(indexed):
    draws = indices()
    conditions, contrasts, background, laps = [], [], [], []
    for d, f in product(DENSITIES, FRACTIONS):
        groups = {m: [[indexed[m, s, d, f, run] for run in range(30)]
                      for s in (SEEDS if m in FAMILIES else (None,))] for m in METHODS}
        arrays = {m: {metric: np.array([[r[metric] for r in group] for group in groups[m]])
                      for metric in METRICS} for m in METHODS}
        for m in METHODS:
            identity = dict(method=m, density_token=d, fraction_token=f)
            for metric in METRICS:
                a = arrays[m][metric]
                stats = seed_stats(a.mean(axis=1).tolist()) if m in FAMILIES else dict(mean=float(a.mean()), seed_means={}, seed_sd=None)
                conditions.append(dict(identity, metric=metric, **stats, road_n=30,
                                       ci95=interval(a.mean(axis=0), draws)))
            lines = [background_line(a, arrays['B1']['flow_background'][0]) for a in arrays[m]['flow_background']]
            background.append(dict(identity, comparator='B1', metric='background_ratio',
                seed_lines=dict(zip(map(str, SEEDS), lines)) if m in FAMILIES else {},
                control_line=lines[0] if m in CONTROLS else None,
                mean=float(np.mean([x['mean'] for x in lines])) if all(x['mean'] is not None for x in lines) else None,
                seed_sd=float(np.std([x['mean'] for x in lines], ddof=1)) if m in FAMILIES and all(x['mean'] is not None for x in lines) else None,
                all_seeds_pass=all(x['screen_ge_095'] for x in lines)))
            lap_values = [[r[LAP] for r in group] for group in groups[m]]
            lines = [lap_line(a) for a in lap_values]
            stats = seed_stats([x['mean'] for x in lines]) if m in FAMILIES else dict(mean=lines[0]['mean'], seed_means={}, seed_sd=None)
            laps.append(dict(identity, **stats, conditional=True, ci95=None,
                completion_rate=sum(x['n'] for x in lines)/(30*len(lines)),
                lines=dict(zip(map(str, SEEDS if m in FAMILIES else ('control',)), lines)),
                paired={b: {str(s): lap_line(a, [r[LAP] for r in groups[b][0]])
                            for s, a in zip(SEEDS if m in FAMILIES else ('control',), lap_values)}
                        for b in ('B1', 'B2')}))
        for left, right in tuple(product(METHODS, ('B2', 'B1'))) + PAIRS:
            delta = road_difference(arrays[left][METRICS[0]], arrays[right][METRICS[0]])
            mean = float(delta.mean())
            bounds = interval(delta, draws)
            contrasts.append(dict(method=left, comparator=right, density_token=d, fraction_token=f,
                                  metric=METRICS[0], mean=mean, ci95=bounds, road_n=30,
                                  sign=interval_sign(bounds),
                                  point_sign=1 if mean > 1e-12 else -1 if mean < -1e-12 else 0))
    return conditions, contrasts, background, laps


def summarize(conditions, contrasts, background, laps):
    summaries = []
    scopes = {'overall': lambda r: True, 'rho_le_040': lambda r: r['density_token'] <= 40,
              'rho_gt_040': lambda r: r['density_token'] > 40,
              **{f'fraction_{f}': lambda r, f=f: r['fraction_token'] == f for f in FRACTIONS}}
    for m, metric, (scope, predicate) in product(METHODS, METRICS, scopes.items()):
        rows = [r for r in conditions if r['method'] == m and r['metric'] == metric and predicate(r)]
        stats = seed_stats([float(np.mean([r['seed_means'][str(s)] for r in rows])) for s in SEEDS]) if m in FAMILIES else dict(mean=float(np.mean([r['mean'] for r in rows])), seed_means={}, seed_sd=None)
        summaries.append(dict(method=m, metric=metric, scope=scope, conditions=len(rows), **stats))
    signs = []
    for a, b in tuple(product(FAMILIES, ('B2', 'B1'))) + PAIRS:
        for scope, predicate in scopes.items():
            rows = [r for r in contrasts if r['method'] == a and r['comparator'] == b and predicate(r)]
            signs.append(dict(method=a, comparator=b, scope=scope, conditions=len(rows), mean=float(np.mean([r['mean'] for r in rows])),
                              positive=sum(r['sign'] == 1 for r in rows), negative=sum(r['sign'] == -1 for r in rows), zero=sum(r['sign'] == 0 for r in rows),
                              point_positive=sum(r['point_sign'] == 1 for r in rows), point_negative=sum(r['point_sign'] == -1 for r in rows),
                              positive_and_all_background_pass=sum(r['sign'] == 1 and next(v['all_seeds_pass'] for v in background if v['method'] == a and v['density_token'] == r['density_token'] and v['fraction_token'] == r['fraction_token']) for r in rows)))
    diagnostics = []
    for m in METHODS:
        rows = [r for r in background if r['method'] == m]
        for s in (SEEDS if m in FAMILIES else ('control',)):
            lines = [r['seed_lines'][str(s)] if m in FAMILIES else r['control_line'] for r in rows]
            diagnostics.append(dict(method=m, seed=s, failures=sum(not r['screen_ge_095'] for r in lines),
                invalid=sum(r['mean'] is None for r in lines), minimum=min((r['mean'] for r in lines if r['mean'] is not None), default=None)))
    return dict(family_seed_summaries=summaries, contrast_summaries=signs, background_by_seed=diagnostics,
                background_all_seeds_failures={m: sum(not r['all_seeds_pass'] for r in background if r['method'] == m) for m in METHODS},
                lap_missing_by_method={m: sum(line['missing'] for r in laps if r['method'] == m for line in r['lines'].values()) for m in METHODS},
                lap_missing_count_scope="logical rows; B1 repeats the same physical road at three fractions",
                lap_missing_physical_by_method={m: sum(line["missing"] for r in laps if r["method"] == m and (m != "B1" or r["fraction_token"] == 90) for line in r["lines"].values()) for m in METHODS},
                illustrative_rho090_f090={name: [r for r in records if r['density_token'] == 90 and r['fraction_token'] == 90]
                                        for name, records in [('conditions', conditions), ('contrasts', contrasts), ('background', background), ('laps', laps)]})


def styles():
    import publication_style as plot_style
    result = {}
    for m, key in zip(CONTROLS, ('baseline1_hdv', 'baseline2_matched', 'noop', 'left_if_safe')):
        result[m] = dict(color=plot_style.COLORS[key], marker=plot_style.MARKERS[key], linestyle=plot_style.LINESTYLES[key], markerfacecolor=plot_style.MARKER_FACE_COLORS[key])
    for m, (color, marker, dash) in zip(FAMILIES, (('#0072B2', 's', '-'), ('#56B4E9', '*', (0, (6, 1, 1, 1))), ('#D55E00', '^', (0, (7, 2))), ('#009E73', 'P', (0, (2, 1))), ('#CC79A7', 'D', (0, (4, 1, 1, 1, 1, 1))))):
        result[m] = dict(color=color, marker=marker, linestyle=dash)
    result['J_G']['markerfacecolor'] = 'white'  # Distinguish from B2's filled magenta diamond.
    return result


def plot_figures(conditions, contrasts, background, laps, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 9,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8, 'pdf.fonttype': 42,
                         'svg.fonttype': 'none', 'axes.linewidth': .65, 'savefig.dpi': 300})
    specs = {'pv_speed': [(conditions, METRICS[0], '', 'mean', 'Скорость PV,\nклеток/шаг')],
             'paired_deltas': [(contrasts, METRICS[0], b, 'mean', f'Δ скорости к {b},\nклеток/шаг') for b in ('B2', 'B1')],
             'flows': [(conditions, 'flow_all', '', 'mean', 'Общий поток,\nавт./(шаг·полосу)'),
                       (conditions, 'flow_background', '', 'mean', 'Фоновый поток,\nавт./(шаг·полосу)'),
                       (background, 'background_ratio', 'B1', 'mean', 'Отношение к B1,\nбезразм.')],
             'stopped_fraction': [(conditions, METRICS[3], '', 'mean', 'Доля времени\nбез движения')],
             'laps': [(laps, '', '', 'mean', 'Время круга при\nзавершении, шаги'),
                      (laps, '', '', 'completion_rate', 'Доля реализаций\nс завершением круга')]}
    audit = {}
    method_styles = styles()
    for name, panels in specs.items():
        height = 1.85*len(panels)+1.35
        fig, axes = plt.subplots(len(panels), 3, figsize=(5.65, height), squeeze=False, sharex=True, sharey='row')
        panel_methods = []
        bands = []
        for i, (records, metric, comparator, field, ylabel) in enumerate(panels):
            lookup = {(r['method'], r['density_token'], r['fraction_token']): r for r in records
                      if r.get('metric', '') == metric and r.get('comparator', '') == comparator}
            for j, f in enumerate(FRACTIONS):
                ax = axes[i, j]
                found = []
                for m in METHODS:
                    series = [lookup[m, d, f] for d in DENSITIES]
                    y = [r[field] if r[field] is not None else np.nan for r in series]
                    x = np.array(DENSITIES)/100
                    line, = ax.plot(x, y, label=('$'+m+'$' if '_' in m else m), linewidth=1.05, markersize=3.3, markeredgewidth=.7, **method_styles[m])
                    if metric in METRICS and field == 'mean':
                        band = ax.fill_between(x, [r['ci95'][0] for r in series], [r['ci95'][1] for r in series], color=line.get_color(), alpha=.09, linewidth=0)
                        bands.append((band, line.get_color()))
                    if name == 'laps' and field == 'mean' and m in FAMILIES:
                        for s in SEEDS:
                            ax.plot(x, [r['seed_means'][str(s)] for r in series], linestyle='none', marker=method_styles[m]['marker'], color=line.get_color(), markersize=1.5)
                    found.append(m)
                panel_methods.append(found)
                ax.axvline(.40, color='#808080', linewidth=.55, linestyle=':', zorder=0)
                if name == 'paired_deltas':
                    ax.axhline(0, color='black', linewidth=.6)
                if metric == 'background_ratio':
                    ax.axhline(.95, color='black', linewidth=.6, linestyle=':')
                ax.set_xlim(.04, .91)
                ax.set_xticks((.1, .3, .5, .7, .9))
                if i == 0:
                    ax.set_title(f'Доля AV = {f/100:.2f}', fontsize=9)
                if j == 0:
                    ax.set_ylabel(ylabel)
                if i == len(panels)-1:
                    ax.set_xlabel('Плотность ρ')
                ax.spines[['top', 'right']].set_visible(False)
                ax.grid(color='#dddddd', linewidth=.45)
                ax.tick_params(length=2.8, width=.6, pad=2)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.55, .015), ncol=3, frameon=False, handlelength=2.5, columnspacing=1.5)
        fig.subplots_adjust(left=.17, right=.985, top=1-.29/height, bottom=1.30/height, wspace=.14, hspace=.22)
        for suffix in ('pdf', 'svg', 'png', 'eps'):
            folder = output_dir / ('eps' if suffix == 'eps' else 'figures')
            folder.mkdir(parents=True, exist_ok=True)
            if suffix == 'eps':
                # EPS cannot encode alpha: preblend the fills against white deliberately.
                from matplotlib.colors import to_rgb
                for band, color in bands:
                    band.set_alpha(1)
                    band.set_facecolor(tuple(.09*c+.91 for c in to_rgb(color)))
            fig.savefig(folder / f'main_{name}.{suffix}', dpi=300)
        audit[name] = dict(width_inches=5.65, panel_methods=panel_methods, legend_entries=9,
                           interval_scope='pointwise 30 shared roads conditional on fixed models',
                           eps_transparency='opaque white-preblended 9% fills')
        plt.close(fig)
    return audit


def tables(summary, winners, output_dir):
    def label(m):
        return '$'+m.replace('_', '_{')+'}$' if '_' in m else m
    def fmt(x):
        return '--' if x is None else f'{x:.4f}'
    def table(name, caption, header, rows):
        path = output_dir / 'tables' / f'main_{name}.tex'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\\begin{table}[htbp]\n\\centering\n\\caption{'+caption+'}\n\\label{tab:main-'+name+'}\n\\small\n\\begin{tabular}{'+ 'l'+'r'*(len(header)-1)+'}\n\\hline\n' + ' & '.join(header)+r' \\'+'\n\\hline\n' + '\n'.join(' & '.join(map(str, r))+r' \\' for r in rows)+'\n\\hline\n\\end{tabular}\n\\end{table}\n')
    overview = [r for r in summary['family_seed_summaries'] if r['metric'] == METRICS[0] and r['scope'] == 'overall']
    table('overview', 'Средняя скорость PV по 54 условиям, клеток/шаг. Для обученных семейств: три отдельных средних и выборочное SD между моделями (не SEM).',
          ['Метод', 'Зерно 14', 'Зерно 15', 'Зерно 16', 'Среднее', 'SD'],
          [[label(r['method'])]+[fmt(r['seed_means'].get(str(s))) for s in SEEDS]+[fmt(r['mean']), fmt(r['seed_sd'])] for r in overview])
    table('contrasts', 'Разность скорости PV по 54 условиям, клеток/шаг, и число условий с положительным ($+$), включающим ноль ($0$) и отрицательным ($-$) условным 95\\%-м дорожным интервалом; численный допуск $10^{-12}$. Среднее по сетке приведено без интервала.',
          ['Контраст', 'Среднее', '$+$', '$0$', '$-$'],
          [[label(r['method'])+' -- '+label(r['comparator']), fmt(r['mean']), r['positive'], r['zero'], r['negative']] for r in summary['contrast_summaries'] if r['scope'] == 'overall'])
    table('background', 'Фоновый поток: среднее парных отношений к B1. Минимум по 54 условиям и число нарушений порога 0.95 для каждой модели; каждый расчёт требует 30 допустимых знаменателей.',
          ['Метод', 'Зерно', 'Минимум $R$', 'Нарушений'],
          [[label(r['method']), str(r['seed'])[-2:] if isinstance(r['seed'], int) else '--', fmt(r['minimum']), r['failures']] for r in summary['background_by_seed']])
    table('selection', 'Итерации фактически выбранных контрольных точек; сохранённые результаты отбора без повторного вычисления.',
          ['Семейство', '20260914', '20260915', '20260916'],
          [[label(m)]+[next(r['candidate']['iteration'] for r in winners if r['family']==m and r['seed']==s) for s in SEEDS] for m in FAMILIES])

