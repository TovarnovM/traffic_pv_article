"""Combined 5-family held-out tables: CPU 10M matrix plus the J_G GPU duplicates.

Per the human decision (2026-09-27) J_G is presented in the SAME tables and
figures as A/P_L/P_G/J_L, with an honest provenance footnote: the four CPU
families come from the CPU 10M matrix, while J_G comes from the GPU-protocol
duplicates (algorithm-state-only non-bitwise resumes, generations 3/5/6) because
the CPU J_G lines stopped at iteration 306. The J_G rows are appended, never
recomputed, and the CPU/GPU training difference is treated as a protocol note,
not as disqualifying.
"""
import json
from pathlib import Path

from sep_research import protocol
from sep_research.export_legacy import json_hash, sha256
from sep_research.eval_selection_10m import tables

FAMILIES = ('A', 'P_L', 'P_G', 'J_L', 'J_G')
CPU_FAMILIES = ('A', 'P_L', 'P_G', 'J_L')
SEEDS = protocol.TRAINING_SEEDS
PROVENANCE = {family: 'cpu_10m_matrix' for family in CPU_FAMILIES}
PROVENANCE['J_G'] = ('gpu_protocol_duplicate_nonbitwise_resume_generations_3_5_6; '
                     'CPU J_G lines stopped at iteration 306')
SCHEMA = 'sep_combined_tables_10m_v1'


def load_rows(path):
    return tables.load_rows(path)


def assemble(cpu_raw, jg_raws):
    rows, sources = [], []
    for path in [cpu_raw, *jg_raws]:
        loaded = load_rows(path)
        rows.extend(loaded)
        sources.append(dict(path=str(path), rows=len(loaded), sha256=sha256(path)))
    return rows, sources


def dedupe_controls(rows):
    """Collapse the per-wave control panels into one, requiring identical rows.

    Every wave measures its own B1/B2/N/L panel on the same frozen grid, so the
    concatenated rows contain one control instance per wave. They are required to
    be byte-identical; the surviving copy is what the tables aggregate.
    """
    unique, seen, duplicates, differing = [], {}, 0, []
    for row in rows:
        if row['method'] in tables.CONTROLS:
            key = (row['method'], row['density_token'], row['fraction_token'], row['run_index'])
            prior = seen.get(key)
            if prior is None:
                seen[key] = row
                unique.append(row)
                continue
            duplicates += 1
            if json_hash(prior) != json_hash(row):
                differing.append(key)
            continue
        unique.append(row)
    return unique, dict(control_duplicates=duplicates, control_differing=len(differing),
                        example_differing=differing[:5])


def verify(rows, *, families=FAMILIES, seeds=SEEDS):
    """Require the full 5-family x 3-seed grid and identical control panels."""
    unique, stats = dedupe_controls(rows)
    learned, controls = tables.index_rows(unique, families=families, seeds=seeds,
                                          control_modes=tables.CONTROLS, runs=tables.RUNS)
    return dict(families=list(families), seeds=list(seeds),
                learned_rows=len(learned), control_instances=len(controls), **stats)


def build(cpu_raw, jg_raws, out_dir):
    rows, sources = assemble(cpu_raw, jg_raws)
    check = verify(rows)
    if check['control_differing']:
        raise ValueError(f"control panels differ across waves: {check['example_differing']}")
    unique, _ = dedupe_controls(rows)
    records = tables.condition_records(unique, families=FAMILIES)
    summary = tables.family_summary(records, families=FAMILIES)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(schema=SCHEMA, families=list(FAMILIES), cpu_families=list(CPU_FAMILIES),
        seeds=list(SEEDS), provenance=PROVENANCE,
        jg_note=('J_G is included in the same set by explicit human decision; its rows are '
                 'GPU-protocol duplicates with non-bitwise resumes (generations 3/5/6) because '
                 'the CPU J_G lines stopped at 306. Reported, not hidden; treated as a protocol '
                 'note rather than a disqualifier.'),
        densities=list(tables.DENSITIES), fractions=list(tables.FRACTIONS), runs=tables.RUNS,
        controls=list(tables.CONTROLS), metrics=list(tables.METRICS),
        sources=sources, rows=len(unique), verification=check,
        interval_scope=('pointwise run-level normal CI for absolute means; paired percentile '
                        'bootstrap CI over 30 runs x 3 seeds'),
        adaptation_notes=tables.ADAPTATION_NOTES)
    (out / 'conditions.json').write_text(json.dumps(dict(meta, records=records), sort_keys=True, indent=1) + '\n')
    (out / 'family_summary.json').write_text(json.dumps(dict(meta, summary=summary), sort_keys=True, indent=1) + '\n')
    tables._write_csv(out / 'conditions.csv', records, tables.CONDITION_FIELDS)
    tables._write_csv(out / 'family_summary.csv', summary, tables.SUMMARY_FIELDS)
    (out / 'SUMMARY.md').write_text(render_summary(meta, records, summary))
    tables.plot_figures(records, unique, out / 'figures', families=FAMILIES)
    return meta, records, summary


def _fmt(value, digits=4):
    return '—' if value is None else f'{value:.{digits}f}'


def render_summary(meta, records, summary):
    lines = ['# Combined held-out test aggregates — 5 families', '',
             'A, P_L, P_G, J_L (CPU 10M matrix) **and J_G** in one set, per explicit human decision.', '',
             '| family | checkpoint provenance |', '|---|---|']
    for family in meta['families']:
        lines.append(f"| {family} | {meta['provenance'][family]} |")
    lines += ['', meta['jg_note'], '',
              '## Primary metric: mean PV speed (equal weight over the 54 conditions)', '',
              '| family | scope | conditions | mean | SEM across conditions | min | max |',
              '|---|---|---:|---:|---:|---:|---:|']
    for row in summary:
        if row['metric'] != tables.PRIMARY or row['scope'].startswith('vs_'):
            continue
        lines.append(f"| {row['family']} | {row['scope']} | {row['conditions']} | {_fmt(row['condition_mean'])} | "
                     f"{_fmt(row['condition_sem'])} | {_fmt(row['condition_min'])} | {_fmt(row['condition_max'])} |")
    lines += ['', '## Mean paired advantage in PV speed vs Baseline2 (equal weight over 54 conditions)', '',
              '| family | scope | conditions | mean Δ vs B2 |', '|---|---|---:|---:|']
    for row in summary:
        if row['metric'] != tables.PRIMARY or row['scope'] != 'vs_B2':
            continue
        lines.append(f"| {row['family']} | {row['scope']} | {row['conditions']} | {_fmt(row['condition_mean'])} |")
    lines += ['', '## Control reference (mean PV speed over 54 conditions)', '',
              '| control | mean |', '|---|---:|']
    for control in tables.CONTROLS:
        values = [r['mean'] for r in records if r['kind'] == 'control' and r['method'] == control
                  and r['metric'] == tables.PRIMARY and r['mean'] is not None]
        lines.append(f"| {control} | {_fmt(sum(values) / len(values)) if values else '—'} |")
    lines += ['', '## Caveats', '',
              '* J_G rows are GPU-protocol duplicates (non-bitwise algorithm-state resumes, generations '
              '3/5/6); the other four families are CPU 10M runs. The CPU/GPU training difference is '
              'treated as a protocol note, not a disqualifier.',
              f"* Every wave measures its own control panel on the same frozen grid; the "
              f"{meta['verification']['control_duplicates']} duplicate control rows were required to be "
              f"byte-identical before aggregation "
              f"({meta['verification']['control_differing']} differed), so the control reference is a "
              f"single 6,480-row panel shared by all five families.",
              '* Seed 20260915 for J_G was appended after the first two seeds; the earlier rows were '
              'never recomputed.',
              '* Intervals are pointwise/descriptive; no multiplicity correction and no formal '
              'non-inferiority claim.',
              '* The frozen 20M anchor and the 12-pair variant gate are unchanged.', '']
    return '\n'.join(lines)
