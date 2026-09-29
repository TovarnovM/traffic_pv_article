"""Validate and plot shipped training summaries; never reads original training logs."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

REVIEW = Path(__file__).resolve().parents[1]
FAMILIES = ('A', 'P_L', 'P_G', 'J_L', 'J_G')
SEEDS = (20260914, 20260915, 20260916)
SELECTED = {'A': (500, 600, 450), 'P_L': (550, 500, 500), 'P_G': (500, 450, 450),
            'J_L': (450, 550, 450), 'J_G': (450, 500, 450)}
COLUMNS = {'family', 'seed', 'iteration', 'env_steps', 'segment', 'source_id',
           'source_row', 'selected_checkpoint', 'pv_speed_mean', 'team_return_mean'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expected_ranges(family, seed):
    if family == 'J_G':
        return {20260914: [(1, 150), (151, 175), (176, 425), (426, 611)],
                20260915: [(1, 150), (151, 611)],
                20260916: [(1, 175), (176, 200), (201, 611)]}[seed]
    if (family, seed) == ('P_G', 20260914):
        return [(1, 306), (307, 575), (576, 611)]
    return [(1, 306), (307, 611)]


def validate(rows, manifest):
    """Fail closed on coverage, lineage, counters, nonfinite values or bad provenance."""
    require(manifest['schema'] == 'training_progress_v1', 'unsupported schema')
    require(manifest['steps_per_iteration'] == 16384 and manifest['source_row_base'] == 1,
            'counter/row convention')
    require(len(rows) == manifest['row_count'] == 9165, 'row count')
    sources = manifest['sources']
    require(len(sources) == 28, 'source count')
    def check_receipts(value):
        if isinstance(value, dict):
            if 'path' in value:
                p = PurePosixPath(value['path'])
                require(not p.is_absolute() and '..' not in p.parts and '\\' not in str(p), 'unsafe source path')
                require(re.fullmatch('[0-9a-f]{64}', value.get('sha256', '')) is not None, 'source hash schema')
            for v in value.values():
                check_receipts(v)
        elif isinstance(value, list):
            for v in value:
                check_receipts(v)
    check_receipts(manifest)
    require(len({s['path'] for s in sources}) == 28, 'duplicate source')
    lines = {(l['family'], l['seed']): l for l in manifest['lines']}
    expected = {(f, s) for f in FAMILIES for s in SEEDS}
    require(len(manifest['lines']) == 15 and set(lines) == expected, 'line coverage')
    grouped = {key: [] for key in expected}
    for r in rows:
        require(set(r) == COLUMNS, 'row schema')
        key = (r['family'], r['seed'])
        require(key in grouped, 'unexpected line')
        for k in ['seed', 'iteration', 'env_steps', 'segment', 'source_id', 'source_row']:
            require(type(r[k]) is int, 'integer required: ' + k)
        require(type(r['selected_checkpoint']) is bool, 'selected flag type')
        require(r['env_steps'] == r['iteration'] * 16384, 'environment counter')
        for k in ['pv_speed_mean', 'team_return_mean']:
            require(type(r[k]) in (float, int) and math.isfinite(r[k]), 'missing/nonfinite metric: ' + k)
        grouped[key].append(r)
    used = set()
    for (family, seed), line in lines.items():
        history = grouped[family, seed]
        require([r['iteration'] for r in history] == list(range(1, 612)), 'iteration coverage/order')
        ranges = expected_ranges(family, seed)
        require([(s['first_iteration'], s['last_iteration']) for s in line['segments']] == ranges,
                'accepted segment ranges')
        selected = SELECTED[family][SEEDS.index(seed)]
        require(line['selected']['iteration'] == selected, 'selected receipt iteration')
        require([r['iteration'] for r in history if r['selected_checkpoint']] == [selected], 'selected markers')
        for index, segment in enumerate(line['segments'], 1):
            require(segment['segment'] == index, 'segment numbering')
            sid = segment['source_id']
            require(type(sid) is int and 0 <= sid < len(sources), 'source id')
            source = sources[sid]
            used.add(sid)
            a, b = ranges[index - 1]
            require(source['rows'] == source['last_iteration'] - source['first_iteration'] + 1, 'source extent')
            require(segment['first_source_row'] == a - source['first_iteration'] + 1, 'source row origin')
            # Bind allowed source ownership, including the two exceptional CPU retries.
            if family == 'J_G':
                suffix = f'jg_gpu/origins/{seed}/train_metrics.jsonl'
            else:
                attempt = '003' if (family, seed) in [('P_L', 20260916), ('P_G', 20260914)] else '002'
                names = ['source/metrics.jsonl', 'metrics.jsonl']
                if (family, seed) == ('P_G', 20260914):
                    names = ['source/source/metrics.jsonl', 'source/partial/metrics.jsonl', 'metrics.jsonl']
                suffix = f'cpu/{family}-{seed}/attempt_{attempt}/' + names[index - 1]
            require(source['path'].endswith('/' + suffix), 'source ownership/excluded attempt')
            for r in history[a - 1:b]:
                require(r['segment'] == index and r['source_id'] == sid, 'segment/source mapping')
                require(r['source_row'] == r['iteration'] - source['first_iteration'] + 1
                        and 1 <= r['source_row'] <= source['rows'], 'source row mapping')
    require(used == set(range(28)), 'unused source')
    return grouped


def load_data(data_dir=REVIEW / 'data/training'):
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / 'manifest.json').read_text())
    require(manifest['data']['file'] == 'progress.jsonl.gz', 'data filename')
    compressed = (data_dir / 'progress.jsonl.gz').read_bytes()
    require(hashlib.sha256(compressed).hexdigest() == manifest['data']['sha256'], 'data checksum')
    payload = gzip.decompress(compressed)
    require(hashlib.sha256(payload).hexdigest() == manifest['data']['uncompressed_sha256'], 'payload checksum')
    require(len(compressed) == manifest['data']['bytes'] and len(payload) == manifest['data']['uncompressed_bytes'], 'data size')
    rows = [json.loads(line) for line in payload.splitlines()]
    validate(rows, manifest)
    return rows, manifest


def plot(rows, manifest, output_dir):
    """One curve artist per accepted segment; no smoothing or bridging restarts."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    grouped = validate(rows, manifest)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = ('#0072B2', '#D55E00', '#009E73')
    styles = ('-', '--', ':')
    settings = {'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                'axes.spines.right': False, 'axes.linewidth': 0.6, 'savefig.dpi': 180,
                'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.hashsalt': 'training-progress-v1'}
    with matplotlib.rc_context(settings):
        for metric, ylabel, title in [
            ('pv_speed_mean', 'PV speed (cells / simulation step)', 'Training priority-vehicle speed'),
            ('team_return_mean', 'Team return (dimensionless episode sum)', 'Training team return')]:
            fig, axes = plt.subplots(2, 3, figsize=(10, 5.7), sharex=True, sharey=True)
            for ax, family in zip(axes.flat, FAMILIES):
                for seed, color, style in zip(SEEDS, colors, styles):
                    history = grouped[family, seed]
                    for index, (a, b) in enumerate(expected_ranges(family, seed)):
                        segment = history[a - 1:b]
                        ax.plot([r['env_steps'] / 1e6 for r in segment], [r[metric] for r in segment],
                                color=color, linestyle=style, linewidth=0.65)
                        if index:
                            # Small seed-colored ticks mark restart positions at the top.
                            boundary = (a - 1) * 16384 / 1e6
                            lane = 0.96 - SEEDS.index(seed) * 0.035
                            ax.plot([boundary, boundary], [lane - .025, lane],
                                    transform=ax.get_xaxis_transform(), color=color, linewidth=1.1)
                            ax.plot(segment[0]['env_steps'] / 1e6, segment[0][metric],
                                    marker='o', markersize=3, markerfacecolor='white',
                                    markeredgecolor=color, markeredgewidth=.7)
                    chosen = next(r for r in history if r['selected_checkpoint'])
                    ax.plot(chosen['env_steps'] / 1e6, chosen[metric], marker='D', markersize=4,
                            markerfacecolor=color, markeredgecolor='black', markeredgewidth=.4)
                ax.set_title(family.replace('_', ' '))
                ax.grid(axis='y', color='#dddddd', linewidth=.4)
                ax.set_xlim(0, 10.1)
                ax.set_xticks([0, 2, 4, 6, 8, 10])
                ax.tick_params(labelbottom=True)
            axes.flat[-1].axis('off')
            handles = [Line2D([], [], color=c, linestyle=s, label=str(seed))
                       for seed, c, s in zip(SEEDS, colors, styles)]
            handles += [Line2D([], [], color='black', marker='D', linestyle='none', label='Selected checkpoint'),
                        Line2D([], [], color='black', marker='o', markerfacecolor='white', linestyle='none', label='First point after resume'),
                        Line2D([], [], color='black', marker='|', linestyle='none', label='Resume boundary (top ticks)')]
            axes.flat[-1].legend(handles=handles, loc='center', frameon=False, title='Training seed / markers', fontsize=8)
            fig.suptitle(title + ' — saved rolling episode summaries', fontsize=11)
            fig.supxlabel('Accepted environment steps (millions)', y=.055)
            fig.supylabel(ylabel, x=.015)
            fig.text(.5, .012, 'Training data, not held-out evaluation. Segments are separate; no confidence bands or additional smoothing.',
                     ha='center', fontsize=8)
            fig.tight_layout(rect=(.025, .08, 1, .95))
            for extension in ('pdf', 'svg', 'png'):
                metadata = {'CreationDate': None, 'ModDate': None} if extension == 'pdf' else ({'Date': None} if extension == 'svg' else {})
                fig.savefig(output_dir / f'{metric}.{extension}', metadata=metadata)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=REVIEW / 'data/training')
    parser.add_argument('--output-dir', type=Path, default=REVIEW / 'build/training')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    rows, manifest = load_data(args.data_dir)
    if not args.validate_only:
        plot(rows, manifest, args.output_dir)
    print(f'Validated {len(rows)} rows: 15 histories, 611 iterations each, 34 accepted segments.')


if __name__ == '__main__':
    main()
