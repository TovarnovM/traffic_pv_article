"""J_G GPU-duplicate sub-panel: validation-seed checkpoint selection (2 seeds).

Explicitly separated from the 12-line CPU matrix:

* these are the two COMPLETED GPU-protocol duplicate J_G runs (training seeds
  20260914 and 20260916) from host pc-413-00u; the third seed 20260915 is still
  training and is absent from every result produced here;
* the runs are algorithm-state-only segmented resumes, explicitly NOT bitwise
  (seed 20260914 = 5 resume generations, seed 20260916 = 3), and both carry
  eligible_for_main_completeness=false, so their numbers are NOT directly
  comparable and must be reported separately;
* the validation protocol and the 4.5 selection rule are identical to the CPU
  variant (frozen root-610003 20-run manifest, rho 0.30 / AV 1.00, deterministic
  argmax, rejected-action guard, <0.05 tie-break on 50 fresh-seed runs, earlier
  checkpoint on a remaining tie).

The frozen 20M anchor, admission.py, evaluate.py and protocol.py are untouched;
`evaluate.prepare_plan` is not used because it requires all three training seeds
per family, so this module builds the same `prepared` mapping directly.
"""
import json
from pathlib import Path

from sep_research import checkpoints, evaluate, protocol
from sep_research.export_legacy import ROOT, json_hash, sha256
from sep_research.eval_selection_10m import selection, test_gate

FAMILY = 'J_G'
ALL_SEEDS = (20260914, 20260915, 20260916)
SEEDS = (20260914, 20260916)          # materialized by default; seed15 appended later
MISSING_SEED = 20260915               # kept for the 2-seed artifacts; now materialized
CANDIDATES = selection.CANDIDATES
CANDIDATE_NAMES = {450: 'periodic_450', 500: 'periodic_500', 550: 'periodic_550',
                   600: 'periodic_600', 611: 'final_611_10010624'}
GENERATIONS = {20260914: 5, 20260915: 6, 20260916: 3}
TARGET_ITERATION = 611
BUDGET = 10_000_000
EXPECTED_STEPS = 10_010_624
SCHEMA_RUN = 'sep_jg_gpu_validated_v1'
SCHEMA_SELECTION = 'sep_selection_jg_gpu_v1'
SCIENCE_STATUS = 'jg_gpu_duplicate_subpanel_separate_from_cpu_matrix'
SUBPANEL_ROOT = 'sep_research/eval_selection_10m_20260925/jg_gpu'


def pairs(seeds=SEEDS):
    return tuple((FAMILY, seed) for seed in seeds)


def pending_seeds(materialized):
    return tuple(seed for seed in ALL_SEEDS if seed not in set(materialized))


def _read(path):
    return evaluate._read_json(path)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _root(root):
    return Path(root) if root is not None else ROOT / SUBPANEL_ROOT


def origin_receipt(seed, *, root=None):
    """Fresh receipt for one completed GPU J_G duplicate run."""
    base = _root(root)
    run = base / 'origins' / str(seed)
    config = _read(run / 'run_config.json')
    contract = config['contract']
    _require(contract['family'] == FAMILY and contract['seed'] == seed, 'J_G family/seed mismatch')
    _require(contract['profile'] == 'main' and contract['budget'] == BUDGET, 'J_G budget/profile mismatch')
    _require(contract['pv_encoder'] == 'graph_mlp'
             and contract['trainable_ids'] == ['pv_policy', 'av_policy'], 'J_G role/encoder mismatch')
    _require(config['status'] == 'budget_reached' and config['iteration'] == TARGET_ITERATION
             and config['final_counters'] == dict(iteration=TARGET_ITERATION, num_env_steps=EXPECTED_STEPS),
             'J_G run did not finish the 10M budget')
    _require(config['finalization'] == dict(algorithm_stop='complete', ray_shutdown='complete',
                                            status='complete', errors=[]), 'J_G finalization incomplete')
    _require(config.get('eligible_for_main_completeness') is False,
             'GPU duplicate must remain ineligible for main completeness')
    _require(len(config.get('resume_segments', [])) == GENERATIONS[seed],
             'unexpected J_G resume generation count')
    for name, digest in contract['sources'].items():
        source = (ROOT / name).resolve()
        _require(source.is_relative_to(ROOT) and sha256(source) == digest,
                 f'J_G contract source identity mismatch: {name}')
    evidence = sorted(path for path in run.glob('completion_evidence*.json') if path.is_file())
    _require(len(evidence) == 1, 'exactly one J_G completion evidence file required')
    completion = _read(evidence[0])
    _require(completion['schema'] == 'sep_completion_evidence_v1'
             and completion['status'] == 'budget_reached' and completion['budget_reached'] is True
             and completion['finalization'] == 'complete' and completion['resumed'] is True
             and completion['eligible_for_main_completeness'] is False,
             'J_G completion evidence mismatch')
    for name, digest in completion['files'].items():
        target = run / name
        _require(target.is_file() and sha256(target) == digest, f'J_G completion payload changed: {name}')
    files = {name: sha256(run / name) for name in
             ('run_config.json', evidence[0].name)}
    candidates = []
    for iteration in CANDIDATES:
        path = base / 'checkpoints' / str(seed) / CANDIDATE_NAMES[iteration]
        manifest = checkpoints.validate_bundle(path, contract)
        _require(manifest['iteration'] == iteration
                 and manifest['num_env_steps'] == iteration * 16384
                 and manifest['budget_reached'] is (iteration == TARGET_ITERATION)
                 and manifest['eligible_for_main_completeness'] is False
                 and manifest['contract'] == contract,
                 f'J_G candidate {iteration} counters/contract mismatch')
        candidates.append(dict(iteration=iteration, path=str(path),
            run_config=str(run / 'run_config.json'), contract_source='gpu_duplicate_run_contract',
            contract_sha256=json_hash(contract), manifest_sha256=sha256(path / 'manifest.json'),
            joint_checkpoint_id=manifest['joint_checkpoint_id'], num_env_steps=manifest['num_env_steps'],
            num_agent_steps=2 * manifest['num_env_steps'], label=manifest['label'],
            role_states=manifest['role_states'],
            initial_role_fingerprints=manifest['initial_role_fingerprints']))
    _require(len({c['manifest_sha256'] for c in candidates}) == len(CANDIDATES), 'duplicate J_G candidate')
    return dict(schema=SCHEMA_RUN, panel='jg_gpu_duplicate_subpanel', family=FAMILY, seed=seed,
        run_dir=str(run), contract=contract, contract_sha256=json_hash(contract),
        files=files, budget=BUDGET, target_iteration=TARGET_ITERATION,
        candidates=candidates, initial_role_fingerprints=config['initial_role_fingerprints'],
        lineage=dict(host='pc-413-00u', generations=GENERATIONS[seed],
                     resume_segments=len(config['resume_segments']),
                     bitwise=False, eligible_for_main_completeness=False,
                     note='GPU-protocol duplicate; algorithm-state-only segmented resume'),
        missing_seed=MISSING_SEED)


def receipts(*, root=None, seeds=SEEDS):
    rows = [origin_receipt(seed, root=root) for seed in seeds]
    paths = [c['path'] for row in rows for c in row['candidates']]
    _require(len(set(paths)) == len(paths), 'duplicate J_G candidate path')
    return rows


def publish_receipts(root=None, destination=None):
    base = _root(root)
    rows = receipts(root=root)
    target = Path(destination) if destination else base / 'origin_receipts.json'
    write = json.dumps(dict(schema=SCHEMA_RUN, panel='jg_gpu_duplicate_subpanel', missing_seed=MISSING_SEED,
                            receipts=rows), sort_keys=True, indent=1) + '\n'
    if target.exists() and target.read_text() != write:
        raise ValueError('existing J_G receipts differ')
    target.write_text(write)
    return rows


def validation_plans(rows):
    manifest = [row._asdict() for row in selection.validation_seed_manifest()]
    by_pair = {(row['family'], row['seed']): row for row in rows}
    seeds = tuple(sorted({row['seed'] for row in rows}))
    _require(seeds and set(by_pair) == set(pairs(seeds)),
             'J_G plans require exactly the supplied completed seeds')
    plans = {}
    for iteration in CANDIDATES:
        bundles = []
        for seed in seeds:
            origin = by_pair[(FAMILY, seed)]
            candidate = next(c for c in origin['candidates'] if c['iteration'] == iteration)
            bundles.append(dict(family=FAMILY, training_seed=seed, bundle=candidate['path'],
                                run_config=candidate['run_config']))
        plans[f'{FAMILY}:{iteration}'] = dict(purpose='validation', seed_manifest=manifest,
            physical=dict(evaluate.PHYSICAL_DEFAULTS), fixed_modes=['B2'], diagnostics=[],
            bundles=bundles)
    return plans


def prepare_plan(plan, rows):
    """Build the evaluate-compatible prepared mapping for one J_G validation plan."""
    by_pair = {(row['family'], row['seed']): row for row in rows}
    physical = dict(evaluate.PHYSICAL_DEFAULTS)
    _require(plan['physical'] == physical and plan['fixed_modes'] == ['B2'] and plan['diagnostics'] == [],
             'J_G plan physical/fixed-mode mismatch')
    manifest = tuple(protocol.SeedRow(**row) for row in plan['seed_manifest'])
    protocol.validate_seed_manifest(manifest)
    _require(set(manifest) == set(selection.validation_seed_manifest()),
             'J_G plan must use the frozen 20-run validation manifest')
    checkpoints_map, bundles, model_manifest = {}, [], {}
    common_sources = None
    for item in plan['bundles']:
        family, seed = item['family'], item['training_seed']
        _require((family, seed) in by_pair, 'J_G bundle outside the two completed seeds')
        path, run_path = Path(item['bundle']).resolve(), Path(item['run_config']).resolve()
        contract = json.loads(run_path.read_text())['contract']
        _require(contract['family'] == family and contract['seed'] == seed
                 and contract['profile'] == 'main' and contract['pv_encoder'] == 'graph_mlp'
                 and contract['trainable_ids'] == ['pv_policy', 'av_policy'],
                 'J_G bundle contract binding mismatch')
        for key in evaluate._BINDING_FIELDS:
            _require(contract['env'].get(key, physical[key]) == physical[key],
                     f'J_G checkpoint physical mismatch: {key}')
        for name, digest in contract['sources'].items():
            source = (ROOT / name).resolve()
            _require(source.is_relative_to(ROOT) and sha256(source) == digest,
                     f'J_G checkpoint source identity mismatch: {name}')
        if common_sources is not None:
            _require(contract['sources'] == common_sources, 'mixed J_G checkpoint sources')
        common_sources = contract['sources']
        _, model_receipt = evaluate.load_pair(path, contract)
        checkpoint_id = model_receipt['joint_checkpoint_id']
        _require(checkpoint_id not in checkpoints_map.values(), 'duplicate J_G pair ID')
        checkpoints_map[(family, seed)] = checkpoint_id
        bundles.append(dict(item, bundle=str(path), run_config=str(run_path), contract=contract))
        model_manifest[f'{family}:{seed}'] = dict(manifest=model_receipt,
            run_config_sha256=sha256(run_path), manifest_sha256=sha256(path / 'manifest.json'))
    checkpoints_map[('B2', None)] = selection._b2_identity(physical)
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
        science_status=SCIENCE_STATUS, heldout_test_ready=False,
        variant=dict(schema=test_gate.SCHEMA_PREPARED, panel='jg_gpu_duplicate_subpanel',
                     families=[FAMILY], seeds=sorted(by_pair_keys := sorted({row['seed'] for row in rows})),
                     missing_seed=None, pending_seeds=list(pending_seeds(by_pair_keys)),
                     gpu_duplicate=True, bitwise=False,
                     generations={str(seed): GENERATIONS[seed] for seed in by_pair_keys},
                     comparable_to_cpu_matrix=False))
    return dict(schema=test_gate.SCHEMA_PREPARED, plan=plan, manifest=manifest,
                checkpoints=checkpoints_map, bundles=bundles, provenance=provenance,
                binding=json_hash(provenance), config_hash=json_hash(physical),
                protocol_hash=sha256(ROOT / 'sep_research/EXPERIMENT_PROTOCOL.md'))


def task_matrix(prepared_documents):
    """One task per evaluate group for each (plan, density, run); 10 x 20 tasks.

    prepared_documents maps plan key -> {"prepared": prepared, "path": Path}.
    """
    tasks = []
    for key, entry in sorted(prepared_documents.items()):
        prepared, path = entry['prepared'], Path(entry['path'])
        for group_key, seed_rows in sorted(evaluate._groups(prepared).items()):
            body = dict(schema=test_gate.SCHEMA_TASK, panel='jg_gpu_duplicate_subpanel',
                        plan_key=key, plan_hash=json_hash(prepared['plan']),
                        binding=prepared['binding'], prepared=str(path),
                        prepared_sha256=sha256(path), density_id=group_key[0],
                        run_index=group_key[1], seed_rows=[row._asdict() for row in seed_rows])
            tasks.append(dict(body, task_id=json_hash(body)))
    _require(len(tasks) == len(CANDIDATES) * 20, 'expected 10 plans x 20 group tasks')
    return tasks


def collect_scores(rows, validation_dirs):
    """Read and verify the 10 J_G validation outputs; return (scores, baseline, outputs)."""
    plans = validation_plans(rows)
    _require(type(validation_dirs) is dict and set(validation_dirs) == set(plans),
             'exact J_G validation output keys required')
    by_pair = {(row['family'], row['seed']): row for row in rows}
    seeds = tuple(sorted({row['seed'] for row in rows}))
    expected = 20 * (1 + len(seeds))
    scores, baseline, outputs = [], None, {}
    for key, plan in plans.items():
        prepared = prepare_plan(plan, rows)
        out = Path(validation_dirs[key])
        _require(out.is_dir() and not out.is_symlink(), 'unsafe J_G validation output')
        read = test_gate.read_prepared_output(out, prepared=prepared)
        b2 = sorted((row for row in read if row['method'] == 'B2'), key=lambda row: row['run_index'])
        _require(len(read) == expected and len(b2) == 20,
                 f'J_G validation requires {expected} rows (20 B2 + {len(seeds)}x20)')
        if baseline is None:
            baseline = b2
        _require(json_hash(baseline) == json_hash(b2), 'repeated J_G B2 panel differs')
        outputs[key] = dict(path=str(out), binding_sha256=prepared['binding'],
                            rows_sha256=json_hash(read))
        iteration = int(key.split(':')[1])
        for seed in seeds:
            origin = by_pair[(FAMILY, seed)]
            candidate = next(c for c in origin['candidates'] if c['iteration'] == iteration)
            scores.append(selection._score_pair(prepared, read, baseline, candidate, key, origin))
    return scores, baseline, outputs


def select(rows, validation_dirs, *, tiebreak_results=None):
    scores, baseline, outputs = collect_scores(rows, validation_dirs)
    seeds = tuple(sorted({row['seed'] for row in rows}))
    winners = [selection.pair_decision(scores, pair, tiebreak_results) for pair in pairs(seeds)]
    receipt = dict(schema=SCHEMA_SELECTION, panel='jg_gpu_duplicate_subpanel',
        family=FAMILY, seeds=list(seeds), missing_seed=None,
        pending_seeds=list(pending_seeds(seeds)),
        candidate_iterations=list(CANDIDATES), origins=rows, scores=scores, winners=winners,
        validation_outputs=outputs, B2_rows=baseline, B2_rows_sha256=json_hash(baseline),
        lineage={str(seed): dict(generations=GENERATIONS[seed], bitwise=False,
                                 eligible_for_main_completeness=False) for seed in seeds},
        comparability=dict(comparable_to_cpu_matrix=False,
            reason='GPU-protocol duplicates with algorithm-state-only non-bitwise resumes; '
                   'third seed still training'),
        ranking=dict(metric='mean_speed_priority',
            order=['reject_anomalous_candidates', 'mean_speed_priority_desc',
                   'tiebreak_50_runs_fresh_seed_below_0.05', 'earlier_iteration'],
            tiebreak_threshold=selection.TIE_THRESHOLD, tiebreak_root=selection.TIEBREAK_ROOT,
            tiebreak_runs=selection.TIEBREAK_RUNS, bootstrap_samples=10_000,
            bootstrap_seed=protocol.BOOTSTRAP_ROOT, run_indices=list(range(20)),
            rejected_guard=dict(factor=selection.REJECTED_FACTOR, floor=selection.REJECTED_FLOOR)),
        tiebreak_results=tiebreak_results or None)
    json_hash(receipt)
    return receipt


def validate_selection(receipt, rows, *, tiebreak_results=None):
    _require(type(receipt) is dict and receipt['schema'] == SCHEMA_SELECTION, 'J_G selection schema')
    fresh = select(rows, {key: value['path'] for key, value in receipt['validation_outputs'].items()},
                   tiebreak_results=tiebreak_results)
    _require(json_hash(receipt) == json_hash(fresh), 'J_G selection receipt differs from recomputation')
    return fresh


def summary_table(receipt):
    """Per-pair winner table plus plateau metrics for every candidate."""
    rows = []
    scores = receipt['scores']
    seeds = tuple(sorted({winner['seed'] for winner in receipt['winners']}))
    for winner in receipt['winners']:
        group = sorted((row for row in scores if (row['family'], row['seed']) ==
                        (winner['family'], winner['seed'])),
                       key=lambda row: row['candidate']['iteration'])
        means = [row['mean_speed_priority'] for row in group]
        row = dict(panel='jg_gpu_duplicate_subpanel', family=winner['family'], seed=winner['seed'],
            status=winner['status'],
            winner=None if winner['candidate'] is None else winner['candidate']['iteration'],
            winner_mean_speed=None if winner['candidate'] is None else winner['mean_speed_priority'],
            top_two_margin=winner['margin'], plateau_span=max(means) - min(means),
            plateau=(max(means) - min(means)) < selection.TIE_THRESHOLD,
            candidates={str(entry['candidate']['iteration']): entry['mean_speed_priority'] for entry in group},
            rejected_anomaly=[entry['candidate']['iteration'] for entry in group
                              if entry['rejected_anomaly']],
            tie=None if not winner.get('tie') else dict(
                iterations=winner['tie']['iterations'], margin=winner['tie']['margin'],
                status=winner['tie']['status'],
                decision=None if 'decision' not in winner['tie'] else winner['tie']['decision']))
        rows.append(row)
    return dict(schema='sep_jg_gpu_selection_table_v1', seeds=list(seeds),
                missing_seed=None, pending_seeds=list(pending_seeds(seeds)),
                comparable_to_cpu_matrix=False, rows=rows)


SCHEMA_TEST_PLAN = 'sep_jg_gpu_test_plan_v1'
TEST_PLAN_KEY = 'jgtest:main'
TEST_FIXED_MODES = ('B1', 'B2', 'N', 'L')
TEST_MANIFEST_ROOT = protocol.TEST_ROOT
TEST_PENDING_NOTE = (
    'seed 20260915 is PENDING: when it finishes, run its own selection and its own test rows '
    'with its own winner, then append it to the J_G panel. A seed15 winner that scores worse than '
    'the existing two does not invalidate these rows; the family aggregate simply gains a third seed.')


def test_plan(rows, winners):
    """Held-out plan for the ready J_G seeds on the frozen dense test grid."""
    by_pair = {(row['family'], row['seed']): row for row in rows}
    chosen = {(row['family'], row['seed']): row for row in winners}
    seeds = tuple(sorted({row['seed'] for row in rows}))
    _require(set(chosen) == set(pairs(seeds)), 'winners must cover exactly the supplied seeds')
    bundles, matrix = [], []
    for seed in seeds:
        winner = chosen.get((FAMILY, seed))
        _require(winner is not None and winner['candidate'] is not None,
                 f'J_G seed {seed} has no resolved winner')
        iteration = winner['candidate']['iteration']
        candidate = next(c for c in by_pair[(FAMILY, seed)]['candidates'] if c['iteration'] == iteration)
        _require(candidate['manifest_sha256'] == winner['candidate']['manifest_sha256'],
                 'winner differs from its origin candidate')
        bundles.append(dict(family=FAMILY, training_seed=seed, bundle=candidate['path'],
                            run_config=candidate['run_config']))
        matrix.append(dict(seed=seed, iteration=iteration, path=candidate['path'],
                           joint_checkpoint_id=candidate['joint_checkpoint_id']))
    manifest = [row._asdict() for row in protocol.seed_manifest('test', TEST_MANIFEST_ROOT)]
    return dict(schema=SCHEMA_TEST_PLAN, purpose='test', panel='jg_gpu_duplicate_subpanel',
        seed_manifest=manifest, physical=dict(evaluate.PHYSICAL_DEFAULTS),
        fixed_modes=list(TEST_FIXED_MODES), diagnostics=[], bundles=bundles, matrix=matrix,
        missing_seed=MISSING_SEED, pending_seeds=list(pending_seeds(seeds)),
        comparable_to_cpu_matrix=False, appendable=True,
        pending_note=TEST_PENDING_NOTE, selection_sha256=json_hash(winners))


def prepare_test_plan(plan, rows):
    """Build the evaluate-compatible prepared mapping for the J_G held-out plan."""
    by_pair = {(row['family'], row['seed']): row for row in rows}
    plan_seeds = tuple(sorted({item['training_seed'] for item in plan['bundles']}))
    physical = dict(evaluate.PHYSICAL_DEFAULTS)
    _require(plan['purpose'] == 'test' and plan['physical'] == physical
             and list(plan['fixed_modes']) == list(TEST_FIXED_MODES) and plan['diagnostics'] == [],
             'J_G test plan physical/fixed-mode mismatch')
    manifest = tuple(protocol.SeedRow(**row) for row in plan['seed_manifest'])
    protocol.validate_seed_manifest(manifest)
    _require(set(manifest) == set(protocol.seed_manifest('test', TEST_MANIFEST_ROOT)),
             'J_G test plan must use the exact frozen dense grid')
    _require({row.split_root for row in manifest} == {TEST_MANIFEST_ROOT},
             'J_G test rows must never use validation seeds')
    checkpoints_map, bundles, model_manifest = {}, [], {}
    common_sources = None
    for item in plan['bundles']:
        family, seed = item['family'], item['training_seed']
        _require((family, seed) in by_pair, 'J_G test bundle outside the ready seeds')
        path, run_path = Path(item['bundle']).resolve(), Path(item['run_config']).resolve()
        contract = json.loads(run_path.read_text())['contract']
        _require(contract['family'] == family and contract['seed'] == seed
                 and contract['profile'] == 'main' and contract['pv_encoder'] == 'graph_mlp'
                 and contract['trainable_ids'] == ['pv_policy', 'av_policy'],
                 'J_G test bundle contract binding mismatch')
        for key in evaluate._BINDING_FIELDS:
            _require(contract['env'].get(key, physical[key]) == physical[key],
                     f'J_G test checkpoint physical mismatch: {key}')
        for name, digest in contract['sources'].items():
            source = (ROOT / name).resolve()
            _require(source.is_relative_to(ROOT) and sha256(source) == digest,
                     f'J_G test checkpoint source identity mismatch: {name}')
        if common_sources is not None:
            _require(contract['sources'] == common_sources, 'mixed J_G test checkpoint sources')
        common_sources = contract['sources']
        _, model_receipt = evaluate.load_pair(path, contract)
        checkpoint_id = model_receipt['joint_checkpoint_id']
        _require(checkpoint_id not in checkpoints_map.values(), 'duplicate J_G test pair ID')
        checkpoints_map[(family, seed)] = checkpoint_id
        bundles.append(dict(item, bundle=str(path), run_config=str(run_path), contract=contract))
        model_manifest[f'{family}:{seed}'] = dict(manifest=model_receipt,
            run_config_sha256=sha256(run_path), manifest_sha256=sha256(path / 'manifest.json'))
    for method in TEST_FIXED_MODES:
        checkpoints_map[(method, None)] = selection._b2_identity(physical) if method == 'B2' else json_hash(
            dict(method=method, config=physical, rule_detection=30, rule_cooldown=1))
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
        science_status='jg_gpu_duplicate_subpanel_heldout_test', heldout_test_ready=False,
        variant=dict(schema=test_gate.SCHEMA_PREPARED, panel='jg_gpu_duplicate_subpanel',
                     subpanel='heldout_test', scope='jg_gpu_test',
                     families=[FAMILY], seeds=sorted(plan_seeds), missing_seed=None,
                     pending_seeds=list(pending_seeds(plan_seeds)),
                     gpu_duplicate=True, bitwise=False, comparable_to_cpu_matrix=False,
                     appendable=True, pending_note=TEST_PENDING_NOTE))
    return dict(schema=test_gate.SCHEMA_PREPARED, plan=plan, manifest=manifest,
                checkpoints=checkpoints_map, bundles=bundles, provenance=provenance,
                binding=json_hash(provenance), config_hash=json_hash(physical),
                protocol_hash=sha256(ROOT / 'sep_research/EXPERIMENT_PROTOCOL.md'))


def test_task_matrix(prepared, prepared_path, prepared_sha256):
    """540 group tasks for the J_G held-out panel (one per density x run)."""
    tasks = []
    for group_key, seed_rows in sorted(evaluate._groups(prepared).items()):
        body = dict(schema=test_gate.SCHEMA_TASK, panel='jg_gpu_duplicate_subpanel',
                    plan_key=TEST_PLAN_KEY, plan_hash=json_hash(prepared['plan']),
                    binding=prepared['binding'], prepared=str(prepared_path),
                    prepared_sha256=prepared_sha256, density_id=group_key[0],
                    run_index=group_key[1], seed_rows=[row._asdict() for row in seed_rows])
        tasks.append(dict(body, task_id=json_hash(body)))
    _require(len(tasks) == len(protocol.DENSITY_TOKENS) * 30, 'expected 18 x 30 J_G test group tasks')
    return tasks


TEST_METRICS = ('mean_speed_priority', 'flow_all', 'flow_background',
                'stopped_fraction_priority', 'mean_priority_lap_time')
SCHEMA_TEST_TABLES = 'sep_jg_gpu_test_tables_v1'


def test_tables(raw_rows, out_dir, *, counts_path=None):
    """Per-seed and 2-seed J_G held-out aggregates on the dense grid."""
    import csv
    seeds = tuple(sorted({row['training_seed'] for row in raw_rows if row['method'] == FAMILY}))
    full = {}
    for row in raw_rows:
        density, fraction = row['density_token'], row['fraction_token']
        for metric in TEST_METRICS:
            value = row[metric]
            keys = ((row['training_seed'], 'aggregate') if row['training_seed'] is not None
                    else (None, *seeds, 'aggregate'))
            for seed_key in keys:
                full.setdefault((seed_key, density, fraction, row['method'], metric), []).append(value)
    records = []
    for seed in list(seeds) + ['aggregate']:
        for density, fraction in [(t, f) for t in protocol.DENSITY_TOKENS for f in protocol.FRACTION_TOKENS]:
            for metric in TEST_METRICS:
                learned = full.get((seed, density, fraction, FAMILY, metric))
                if not learned:
                    continue
                from sep_research.analysis import absolute_summary, paired_summary
                summary = absolute_summary(learned)
                records.append(dict(panel='jg_gpu_duplicate_subpanel', seed=seed, kind='absolute',
                    density_token=density, fraction_token=fraction, metric=metric,
                    n=summary['valid_runs'], mean=summary['mean'], sem=summary['sem'],
                    ci_low=None if summary['ci95'] is None else summary['ci95'][0],
                    ci_high=None if summary['ci95'] is None else summary['ci95'][1]))
                for control in TEST_FIXED_MODES:
                    base = full.get((seed, density, fraction, control, metric))
                    if base is None:
                        base = full.get((None, density, fraction, control, metric))
                    if base and len(base) != len(learned) and len(base) * len(seeds) == len(learned):
                        base = base * len(seeds)
                    if not base or len(base) != len(learned):
                        continue
                    paired = paired_summary(learned, base, samples=10_000, seed=protocol.BOOTSTRAP_ROOT,
                                            allow_missing=metric == 'mean_priority_lap_time')
                    records.append(dict(panel='jg_gpu_duplicate_subpanel', seed=seed, kind='paired',
                        comparator=control, density_token=density, fraction_token=fraction,
                        metric=metric, n=paired['valid_pairs'], mean=paired['mean'], sem=None,
                        ci_low=None if paired['ci95'] is None else paired['ci95'][0],
                        ci_high=None if paired['ci95'] is None else paired['ci95'][1]))
    summary = []
    for seed in list(seeds) + ['aggregate']:
        for metric in TEST_METRICS:
            for control in [None] + list(TEST_FIXED_MODES):
                subset = [r for r in records if r['seed'] == seed and r['metric'] == metric
                          and r.get('comparator') == control and r['mean'] is not None]
                if not subset:
                    continue
                means = [r['mean'] for r in subset]
                summary.append(dict(seed=seed, metric=metric,
                    comparator=control or '', kind='absolute' if control is None else 'paired',
                    conditions=len(means), mean=sum(means) / len(means),
                    min=min(means), max=max(means)))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(schema=SCHEMA_TEST_TABLES, panel='jg_gpu_duplicate_subpanel', subpanel='heldout_test',
                seeds=list(seeds), missing_seed=None, pending_seeds=list(pending_seeds(seeds)),
                comparable_to_cpu_matrix=False,
                levels=['per_seed', 'two_seed_aggregate'], densities=list(protocol.DENSITY_TOKENS),
                fractions=list(protocol.FRACTION_TOKENS), runs=30,
                controls=list(TEST_FIXED_MODES), metrics=list(TEST_METRICS), rows=len(raw_rows),
                counts=json.loads(Path(counts_path).read_text()) if counts_path else None,
                pending_note=TEST_PENDING_NOTE)
    (out / 'conditions.json').write_text(json.dumps(dict(meta, records=records), sort_keys=True, indent=1) + '\n')
    (out / 'summary.json').write_text(json.dumps(dict(meta, summary=summary), sort_keys=True, indent=1) + '\n')
    fields = ('panel', 'seed', 'kind', 'comparator', 'density_token', 'fraction_token', 'metric',
              'n', 'mean', 'sem', 'ci_low', 'ci_high')
    with (out / 'conditions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    with (out / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('seed', 'metric', 'kind', 'comparator',
                                                    'conditions', 'mean', 'min', 'max'))
        writer.writeheader()
        for record in summary:
            writer.writerow(record)
    lines = ['# J_G GPU-duplicate held-out test aggregates (2 ready seeds)', '',
             'Panel `jg_gpu_duplicate_subpanel`: GPU-protocol duplicates, algorithm-state-only',
             'non-bitwise resumes (seed 20260914 = 5 generations, seed 20260916 = 3), both',
             'eligible_for_main_completeness=false; **NOT comparable to the 12-line CPU matrix**.',
             ('All three seeds are present; the panel was built append-only (seed 20260915 was added '
              'last and the earlier two seeds were never recomputed).'
              if not pending_seeds(seeds) else
              f"Seed(s) {list(pending_seeds(seeds))} are PENDING and absent from every number here. "
              'When they finish, run their own selection and their own rows and append them: a later '
              'winner that scores worse does not invalidate this data, the family aggregate simply '
              'gains another seed.'), '',
             'Grid: 18 densities (0.05..0.90) x AV 0.90/0.95/1.00 x 30 runs; controls B1/B2/N/L;',
             '2000-step horizon. Levels: each seed separately and the 2-seed aggregate (60 run-level',
             'observations).', '',
             '| seed | mean PV speed (all conditions) | vs B2 | vs B1 | vs N | vs L |',
             '|---|---:|---:|---:|---:|---:|']
    def cell(seed, comparator):
        found = next((r for r in summary if r['seed'] == seed and r['metric'] == 'mean_speed_priority'
                      and r['comparator'] == comparator), None)
        return '—' if found is None else f"{found['mean']:.4f}"
    for seed in list(seeds) + ['aggregate']:
        lines.append(f"| {seed} | {cell(seed, '')} | {cell(seed, 'B2')} | {cell(seed, 'B1')} | "
                     f"{cell(seed, 'N')} | {cell(seed, 'L')} |")
    lines += ['', 'Caveats: GPU protocol, non-bitwise lineage, third seed pending, no multiplicity',
              'correction; intervals are pointwise/descriptive.', '']
    (out / 'SUMMARY.md').write_text('\n'.join(lines))
    return meta, records, summary
