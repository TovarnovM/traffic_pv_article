"""Labelled 12-pair held-out TEST gate (J_G explicitly deferred).

The frozen 20M test gate (`evaluate._test_inputs`, `evaluate._check_test_prepared`
and the anchored `manifests/main_profile.json`) requires all 15 anchored pairs and
is left byte-identical. This module is the declared intermediate variant:

* it accepts only the published `sep_selection_10m_v1` receipt with all 12 winners
  resolved (tie-break aware) and the canonical `sep_test_plan_10m_v1` plan it
  produced;
* it revalidates every selected bundle against its origin contract and the frozen
  dense test grid (root 710009, rho 0.05..0.90 step 0.05, AV 0.90/0.95/1.00, 20 runs);
* it builds the same `prepared` mapping that `evaluate.run_group`, `evaluate._worker_init`
  and `evaluate._validate_output_rows` consume, so execution reuses the frozen
  evaluator unchanged;
* `read_test_10m` mirrors `evaluate.read_evaluation` without the 20M anchor branch.

No frozen file is imported for writing and none is modified.
"""
import json
import os
from pathlib import Path
import tempfile

from research import evaluate_pv_graph_ppo as graph_eval
from sep_research import checkpoints, evaluate, protocol
from sep_research.export_legacy import ROOT, json_hash, sha256
from sep_research.eval_selection_10m import selection

SCHEMA_PREPARED = 'sep_test10m_prepared_v1'
SCHEMA_TASK = 'sep_test10m_task_v1'
PLAN_KEY = 'test:main'
FIXED_MODES = ('B1', 'B2', 'N', 'L')
SCIENCE_STATUS = 'intermediate_12pair_test_matrix_jg_deferred'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    return evaluate._read_json(path)


def prepared_document(prepared):
    """JSON-safe form of the prepared mapping (SeedRow tuple and tuple keys -> lists)."""
    document = {key: value for key, value in prepared.items() if key != 'manifest'}
    document['schema'] = SCHEMA_PREPARED
    document['manifest'] = [row._asdict() for row in prepared['manifest']]
    document['checkpoints'] = [[list(key), value] for key, value in prepared['checkpoints'].items()]
    return document


def load_prepared(path):
    document = _read(path)
    _require(document.get('schema') == SCHEMA_PREPARED, 'prepared document schema')
    document['manifest'] = tuple(protocol.SeedRow(**row) for row in document.pop('manifest'))
    document['checkpoints'] = {(key[0], key[1]): value for key, value in document.pop('checkpoints')}
    return document


def _verify_selection_binding(plan, receipts, receipt):
    """Bind every planned bundle to a resolved winner of the 12-pair receipt."""
    _require(receipt.get('schema') == selection.SCHEMA_SELECTION, 'selection schema')
    _require(plan.get('selection_sha256') == json_hash(receipt), 'plan is not bound to this receipt')
    winners = {(row['family'], row['seed']): row for row in receipt['winners']}
    _require(set(winners) == set(selection.pairs()), 'receipt must cover the 12 pairs')
    origins = {(row['family'], row['seed']): row for row in receipts}
    _require(set(origins) == set(selection.pairs()), 'origin receipts must cover the 12 pairs')
    seen = {}
    for item in plan['bundles']:
        key = (item['family'], item['training_seed'])
        _require(key in winners and key in origins, 'bundle outside the 12-pair matrix')
        winner = winners[key]
        _require(winner['status'] in ('selected', 'selected_after_tiebreak')
                 and winner['candidate'] is not None, f'unresolved winner {key}')
        chosen = winner['candidate']
        origin_candidate = next((c for c in origins[key]['candidates']
                                 if c['iteration'] == chosen['iteration']), None)
        _require(origin_candidate is not None, 'winner absent from its origin receipt')
        _require(item['bundle'] == chosen['path'] and item['run_config'] == origin_candidate['run_config'],
                 'planned bundle/run_config differ from the selected candidate')
        _require(sha256(Path(item['bundle']) / 'manifest.json') == chosen['manifest_sha256'],
                 'selected bundle manifest changed')
        seen[json_hash(item)] = chosen['joint_checkpoint_id']
    _require(len(seen) == len(selection.pairs()), 'plan must contain exactly the 12 selected bundles')


def prepare_test_plan_10m(plan, *, receipts, selection_receipt, deep_selection=True, smoke=False):
    """Validate the 12-pair held-out plan and build the evaluate-compatible prepared map.

    smoke=True admits a declared single-group subset of the frozen grid for the
    machinery smoke test only; the scientific status is then marked as a smoke.
    """
    from sep_research.eval_selection_10m import selection as sel
    receipt = selection_receipt
    if deep_selection:
        fresh = sel.validate_selection(receipt, receipts=receipts,
                                       tiebreak_results=receipt.get('tiebreak_results'))
        _require(json_hash(fresh) == json_hash(receipt), 'selection receipt differs from recomputation')
    if not smoke:
        canonical = sel._test_inputs(receipt)
        _require(json_hash(plan) == json_hash(canonical), 'plan differs from the canonical 12-pair plan')
    _require(plan.get('schema') == sel.SCHEMA_TEST and plan.get('purpose') == 'test',
             'plan schema/purpose mismatch')
    _require(plan.get('fixed_modes') == list(FIXED_MODES) and plan.get('diagnostics') == [],
             'fixed modes/diagnostics mismatch')
    _verify_selection_binding(plan, receipts, receipt)

    physical = dict(evaluate.PHYSICAL_DEFAULTS)
    _require(plan['physical'] == physical, 'test physical config differs from the frozen evaluator defaults')
    _require(physical['warmup_steps'] == 1000 and physical['measure_steps'] == 2000,
             'anchored held-out horizon mismatch')

    manifest = tuple(protocol.SeedRow(**row) for row in plan['seed_manifest'])
    protocol.validate_seed_manifest(manifest)
    _require(all(row.split == 'test' and row.split_root == protocol.TEST_ROOT for row in manifest),
             'manifest split/root mismatch')
    frozen = set(protocol.seed_manifest('test', protocol.TEST_ROOT))
    _require(set(manifest) <= frozen, 'manifest outside the frozen dense grid')
    if smoke:
        _require(1 <= len({(row.canonical_density_id, row.run_index) for row in manifest}) <= 2,
                 'smoke plan must contain at most two groups')
    else:
        _require(set(manifest) == frozen, 'test requires the exact frozen dense grid')
        _require(len({(row.density_token, row.fraction_token) for row in manifest})
                 == len(protocol.DENSITY_TOKENS) * len(protocol.FRACTION_TOKENS), 'grid not dense')

    checkpoints_map, bundles, model_manifest = {}, [], {}
    common_sources = None
    for item in plan['bundles']:
        _require(set(item) == {'family', 'training_seed', 'bundle', 'run_config'},
                 'explicit bundle, run_config, family and training_seed required')
        family, seed = item['family'], item['training_seed']
        _require(family in selection.FAMILIES and seed in protocol.TRAINING_SEEDS,
                 'bundle outside the 12-pair matrix')
        _require((family, seed) not in checkpoints_map, 'duplicate bundle instance')
        path, run_path = Path(item['bundle']).resolve(), Path(item['run_config']).resolve()
        contract = json.loads(run_path.read_text())['contract']
        encoder = 'graph_mlp' if family.endswith('_G') else 'local_mlp'
        trainable = ['av_policy'] if family == 'A' else ['pv_policy'] if family.startswith('P_') else ['pv_policy', 'av_policy']
        _require(contract['family'] == family and contract['seed'] == seed
                 and contract['pv_encoder'] == encoder and contract['env']['controller_mode'] == family
                 and contract['env']['pv_encoder'] == encoder and contract['trainable_ids'] == trainable,
                 'bundle family/seed/role/encoder binding mismatch')
        _require(contract['profile'] == 'main', 'held-out bundles require main-profile contracts')
        for key in evaluate._BINDING_FIELDS:
            _require(contract['env'].get(key, physical[key]) == physical[key],
                     f'checkpoint physical mismatch: {key}')
        sources = contract['sources']
        _require('sep_research/env_core.py' in sources, 'missing core source identity')
        for name, digest in sources.items():
            source = (ROOT / name).resolve()
            _require(source.is_relative_to(ROOT) and sha256(source) == digest,
                     f'checkpoint source identity mismatch: {name}')
        if common_sources is not None:
            _require(sources == common_sources, 'mixed checkpoint sources')
        common_sources = sources
        _, model_receipt = evaluate.load_pair(path, contract)
        checkpoint_id = model_receipt['joint_checkpoint_id']
        _require(checkpoint_id not in checkpoints_map.values(), 'duplicate pair ID')
        checkpoints_map[(family, seed)] = checkpoint_id
        bundles.append(dict(item, bundle=str(path), run_config=str(run_path), contract=contract))
        model_manifest[f'{family}:{seed}'] = dict(manifest=model_receipt, run_config_sha256=sha256(run_path),
                                                  manifest_sha256=sha256(path / 'manifest.json'))
    _require({b['family'] for b in bundles} == set(selection.FAMILIES),
             'held-out plan must cover the four families')
    for family in {b['family'] for b in bundles}:
        _require({b['training_seed'] for b in bundles if b['family'] == family} == set(protocol.TRAINING_SEEDS),
                 'each family requires all three training seeds')
    for method in FIXED_MODES:
        checkpoints_map[(method, None)] = json_hash(dict(method=method, config=physical,
                                                         rule_detection=30, rule_cooldown=1))

    plan = json.loads(json.dumps(plan, allow_nan=False))
    plan.update(seed_manifest=[row._asdict() for row in manifest], physical=physical, legacy_cache=None)
    provenance = dict(plan=plan, models=model_manifest, legacy=None, deviations={},
        sources={str(path.relative_to(ROOT)): sha256(path) for path in [
            *sorted((ROOT / 'src/snfs_traffic').rglob('*.py')),
            ROOT / 'research/evaluate_pv_graph_ppo.py',
            *(ROOT / 'sep_research' / name for name in ('evaluate.py', 'export_legacy.py', 'env_core.py',
                'observations.py', 'inference.py', 'checkpoints.py', 'protocol.py', 'analysis.py',
                'EXPERIMENT_PROTOCOL.md', 'TECHNICAL_SPEC.md'))]},
        runtime=dict(python=__import__('platform').python_version(), numpy=__import__('numpy').__version__,
                     torch=__import__('torch').__version__),
        science_status=(SCIENCE_STATUS if not smoke else 'development_smoke_not_scientific_results'),
        heldout_test_ready=False,
        variant=dict(schema=SCHEMA_PREPARED, scope='12_pair_jg_deferred', smoke=bool(smoke),
                     selection_sha256=json_hash(receipt),
                     frozen_20m_gate='unmodified; requires 15 anchored pairs',
                     tiebreak_results_bound=receipt.get('tiebreak_results') is not None))
    return dict(plan=plan, manifest=manifest, checkpoints=checkpoints_map, bundles=bundles,
                provenance=provenance, binding=json_hash(provenance), config_hash=json_hash(physical),
                protocol_hash=sha256(ROOT / 'sep_research/EXPERIMENT_PROTOCOL.md'))


def _prepared_digest(prepared):
    return json_hash(prepared_document(prepared))


def task_matrix(prepared, prepared_path, prepared_sha256, *, expected=None):
    """One task per evaluate group (density, run); 360 tasks for the dense grid."""
    tasks = []
    for key, seed_rows in sorted(evaluate._groups(prepared).items()):
        body = dict(schema=SCHEMA_TASK, plan_key=PLAN_KEY, plan_hash=json_hash(prepared['plan']),
                    binding=prepared['binding'], prepared=str(prepared_path),
                    prepared_sha256=prepared_sha256, density_id=key[0], run_index=key[1],
                    seed_rows=[row._asdict() for row in seed_rows])
        tasks.append(dict(body, task_id=json_hash(body)))
    if expected is not None:
        _require(len(tasks) == expected, f'expected {expected} group tasks, got {len(tasks)}')
    return tasks


def smoke_plan(canonical_plan, *, density_token=30, fraction_token=100, run_index=0):
    """Declared single-group subset of the frozen test grid for the smoke test."""
    plan = json.loads(json.dumps(canonical_plan, allow_nan=False))
    rows = [row for row in protocol.seed_manifest('test', protocol.TEST_ROOT)
            if row.density_token == density_token and row.fraction_token == fraction_token
            and row.run_index == run_index]
    _require(rows, 'smoke condition is not in the frozen test grid')
    plan['seed_manifest'] = [row._asdict() for row in rows]
    plan['smoke'] = dict(density_token=density_token, fraction_token=fraction_token, run_index=run_index)
    return plan


def _verify_layout(out, prepared, groups):
    return evaluate._verify_reader_layout(out, prepared, groups)


def _panel_label(prepared):
    """Reporting label/status carried by the prepared variant block."""
    variant = prepared.get('provenance', {}).get('variant', {})
    label = variant.get('panel') or variant.get('scope') or 'prepared_variant'
    return dict(variant=label, panel=variant.get('panel', label),
                science_status=prepared['provenance'].get('science_status', SCIENCE_STATUS))


def read_test_10m(out_dir, *, prepared):
    """Verified read of one complete 12-pair test invocation (no frozen 20M anchor)."""
    _require(prepared['plan'].get('purpose') == 'test', 'held-out plan required')
    return read_prepared_output(out_dir, prepared=prepared)


def read_prepared_output(out_dir, *, prepared):
    """Generic verified read of one complete prepared invocation (no frozen anchor).

    Used by the 12-pair held-out test and by the labelled J_G GPU-duplicate
    validation sub-panel; the frozen 20M reader is untouched.
    """
    if (prepared['plan'] != prepared['provenance']['plan']
            or prepared['config_hash'] != json_hash(prepared['plan']['physical'])
            or prepared['protocol_hash'] != prepared['provenance']['sources']['sep_research/EXPERIMENT_PROTOCOL.md']
            or [row._asdict() for row in prepared['manifest']] != prepared['plan']['seed_manifest']):
        raise ValueError('expected preparation mismatch')
    out = Path(out_dir)
    _require(not out.is_symlink() and out.is_dir(), 'missing or unsafe output directory')
    groups = evaluate._groups(prepared)
    _verify_layout(out, prepared, groups)
    evaluate._verify_provenance(out, prepared)
    chunks = out / 'chunks'
    rows = [row for key in sorted(groups)
            for row in evaluate._read_chunk(chunks / f'{key[0]}_{key[1]}.json', prepared, groups[key])]
    evaluate._validate_output_rows(rows, prepared, prepared['manifest'])
    counts = evaluate._counts(prepared, rows, groups)
    counts.update(_panel_label(prepared), heldout_test_ready=False)
    for name, value in (('raw.json', rows), ('counts.json', counts)):
        if json_hash(_read(out / name)) != json_hash(value):
            raise ValueError(f'existing output mismatch: {name}')
    import csv
    import io
    fields = list(dict.fromkeys(k for row in rows for k in sorted(row)))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: graph_eval._csv_cell(row.get(k)) for k in fields})
    if (out / 'raw.csv').read_bytes() != stream.getvalue().encode('utf-8'):
        raise ValueError('existing CSV mismatch')
    return rows


def finalize_test_output(prepared, chunks_source, out_dir):
    """Publish the verified raw test output from already-verified chunks (no simulation)."""
    out = Path(out_dir)
    if out.exists():
        raise FileExistsError(out)
    groups = evaluate._groups(prepared)
    out.mkdir(parents=True)
    evaluate._atomic_json(out / 'provenance.json', dict(prepared['provenance'], binding=prepared['binding']))
    (out / 'chunks').mkdir()
    expected = {f'{key[0]}_{key[1]}.json' for key in groups}
    source = Path(chunks_source)
    _require({path.name for path in source.iterdir()} == expected, 'chunk set differs from the planned groups')
    for name in sorted(expected):
        os.link(source / name, out / 'chunks' / name)
    rows = [row for key in sorted(groups)
            for row in evaluate._read_chunk(out / 'chunks' / f'{key[0]}_{key[1]}.json', prepared, groups[key])]
    evaluate._validate_output_rows(rows, prepared, prepared['manifest'])
    counts = evaluate._counts(prepared, rows, groups)
    counts.update(_panel_label(prepared), heldout_test_ready=False)
    evaluate._atomic_json(out / 'raw.json', rows)
    evaluate._atomic_json(out / 'counts.json', counts)
    with tempfile.TemporaryDirectory(prefix='.csv.partial_', dir=out) as directory:
        temp = Path(directory) / 'raw.csv'
        graph_eval._write_csv(temp, [{k: row[k] for k in sorted(row)} for row in rows])
        target = out / 'raw.csv'
        if target.exists():
            _require(target.read_bytes() == temp.read_bytes(), 'existing CSV mismatch')
        else:
            os.link(temp, target)
    return rows, counts
