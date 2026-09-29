"""Core of the labelled 10M-matrix selection variant. See package docstring.

Nothing here mutates sep_research.admission, sep_research.evaluate,
sep_research.protocol or the frozen 20M anchor. Validation execution reuses
evaluate.prepare_plan / evaluate.read_evaluation unchanged, so the frozen
validation manifest and reader semantics are the 20M protocol's own.
"""
import json
import math
from pathlib import Path

from sep_research import checkpoints, evaluate, protocol
from sep_research.analysis import absolute_summary, paired_summary
from sep_research.cluster_cpu import contract as cpu_contract
from sep_research.cluster_cpu import runner as cpu_runner
from sep_research.export_legacy import ROOT, json_hash, sha256

FAMILIES = ('A', 'P_L', 'P_G', 'J_L')          # J_G deliberately excluded.
SEEDS = protocol.TRAINING_SEEDS
TARGET_ITERATION = 611
BUDGET = 10_000_000
CANDIDATES = tuple(cpu_contract.candidates(TARGET_ITERATION))   # 450,500,550,600,611
CPU_PROFILE = cpu_contract.production_profile()
CPU_PROFILE_HASH = cpu_contract.validate_profile(CPU_PROFILE)
ARCHIVE = ROOT / 'sep_research/collected_20260924/cpu'

VALIDATION_DENSITIES = (30,)
VALIDATION_FRACTIONS = (100,)
VALIDATION_RUNS = 20
FIXED_MODES = ('B2',)                     # 4.1 baseline2 matched comparator.
TIEBREAK_ROOT = 620000                    # fresh validation seed namespace (4.5);
                                          # verified collision-free against the frozen
                                          # reserved guard and all four split manifests.
TIEBREAK_RUNS = 50
TIE_THRESHOLD = 0.05                      # PV speed units (4.5 step 4)
REJECTED_FACTOR = 3.0                     # 4.5 step 3 guard: mean rejected per run
REJECTED_FLOOR = 1.0                      # never trip below three rejected actions/run

SCHEMA_RUN = 'sep_cpu_validated_10m_v1'
SCHEMA_SELECTION = 'sep_selection_10m_v1'
SCHEMA_TEST = 'sep_test_plan_10m_v1'
EXCLUDED_FAMILIES = ('J_G',)


def pairs():
    return tuple((family, seed) for family in FAMILIES for seed in SEEDS)


def pair_key(family, seed):
    return f'{family}:{seed}'


def validation_key(family, iteration):
    return f'{family}:{iteration}'


def _read(path):
    return evaluate._read_json(path)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _attempt_dirs(family, seed):
    base = ARCHIVE / f'{family}-{seed}'
    _require(base.is_dir(), f'missing archive pair: {family}-{seed}')
    return sorted(path for path in base.iterdir() if path.is_dir())


def origin_attempt(family, seed):
    """The collected attempt holding run_config/result/completion evidence."""
    complete = [path for path in _attempt_dirs(family, seed) if (path / 'result.json').is_file()]
    _require(len(complete) == 1, f'expected exactly one completed attempt for {family}-{seed}')
    return complete[0]


def bundle_attempt(family, seed, iteration):
    """Attempt owning one candidate bundle; may differ from origin_attempt (P_G14)."""
    found = [path for path in _attempt_dirs(family, seed)
             if (path / 'checkpoints' / f'iteration_{iteration:06d}' / 'manifest.json').is_file()]
    _require(len(found) == 1, f'candidate {iteration} not unique for {family}-{seed}')
    return found[0]


def authorized_segment_contract(candidate, origin):
    """True iff a candidate contract differs from the origin only by a segment seed.

    The CPU campaign resumed P_G-20260914 at iteration 575 (authorized
    interrupted_continuation_10m). Its 450/500/550 bundles were written by the
    killed predecessor segment, so their continuation/env.seed/rllib.seed
    identify that segment. Everything scientific (sources, models, spaces, PPO,
    env geometry) must be byte-identical to the origin contract.
    """
    sections = {k for k in set(candidate) | set(origin) if candidate.get(k) != origin.get(k)}
    if not sections:
        return False
    if sections - {'continuation', 'env', 'rllib'}:
        return False
    left, right = candidate.get('continuation'), origin.get('continuation')
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    # Only the authorised segment boundary and its derived seed may differ.
    if ({k: v for k, v in left.items() if k not in ('segment_seed', 'start_iteration')}
            != {k: v for k, v in right.items() if k not in ('segment_seed', 'start_iteration')}):
        return False
    if type(left.get('start_iteration')) is not int or type(right.get('start_iteration')) is not int:
        return False
    if not left['start_iteration'] < right['start_iteration']:
        return False
    seed = candidate['env']['seed']
    if left.get('segment_seed') != seed:
        return False
    for section, allowed in (('env', {'seed'}), ('rllib', {'seed', 'env_config'})):
        changed = {k for k in set(candidate[section]) | set(origin[section])
                   if candidate[section].get(k) != origin[section].get(k)}
        if changed - allowed:
            return False
    nested = {k for k in set(candidate['rllib']['env_config']) | set(origin['rllib']['env_config'])
              if candidate['rllib']['env_config'].get(k) != origin['rllib']['env_config'].get(k)}
    return (candidate['rllib'].get('seed') == seed and not nested - {'seed'}
            and candidate['rllib']['env_config'].get('seed') == seed)


def validate_cpu_run(run_dir, *, deep=True):
    """Return the validated archived CPU result; never publishes anything.

    deep=True uses cluster_cpu.runner.validate_result (status, completion
    payload hashes, plan/recovery chain, checkpoint identities, seed audit,
    limits). deep=False performs the schema/status/contract subset only and is
    for tests with synthetic evidence.
    """
    run = Path(run_dir).resolve()
    result = _read(run / 'result.json')
    if deep:
        return cpu_runner.validate_result(run, run_id=result['row']['run_id'])
    _require(result.get('schema') == 'cpu_result_v2' and result.get('status') == 'completed',
             'cpu result not completed')
    return result


ORIGIN_FILES = ('result.json', 'plan.json', 'completion.json', 'metrics.jsonl', 'seed_audit.json')


def origin_receipt(family, seed, *, deep=True, run_config_root=None):
    """Build one fresh, unpublishable origin receipt schema sep_cpu_validated_10m_v1."""
    run = origin_attempt(family, seed)
    result = validate_cpu_run(run, deep=deep)
    contract = result['contract']
    _require(contract['family'] == family and contract['seed'] == seed, 'origin family/seed mismatch')
    _require(contract['profile'] == 'main' and contract['budget'] == BUDGET, 'origin budget/profile mismatch')
    _require(result['profile_hash'] == CPU_PROFILE_HASH, 'origin CPU profile hash mismatch')
    _require(result['final_iteration'] == TARGET_ITERATION
             and result['endpoint']['iteration'] == TARGET_ITERATION, 'origin target iteration mismatch')
    files = {name: sha256(run / name) for name in ORIGIN_FILES}
    candidates, attempts, notes = [], {}, []
    for iteration in CANDIDATES:
        attempt = bundle_attempt(family, seed, iteration)
        attempts[str(iteration)] = attempt.name
        path = attempt / 'checkpoints' / f'iteration_{iteration:06d}'
        declared = _read(path / 'manifest.json')['contract']
        if json_hash(declared) == json_hash(contract):
            expected, source = contract, 'origin_contract'
        elif authorized_segment_contract(declared, contract):
            expected, source = declared, 'authorized_segment_contract'
            notes.append(f'candidate {iteration} belongs to {attempt.name} segment '
                         f"seed={declared['continuation']['segment_seed']} "
                         f"start={declared['continuation']['start_iteration']}")
        else:
            raise ValueError(f'candidate {family}-{seed} iteration {iteration} contract differs '
                             'from the admitted origin contract')
        manifest = checkpoints.validate_bundle(path, expected)
        pair = manifest['joint_checkpoint_id']
        _require(manifest['iteration'] == iteration and manifest['num_env_steps'] == iteration * cpu_contract.BATCH,
                 'candidate counters mismatch')
        _require(manifest['initial_role_fingerprints'] == result['initial_role_fingerprints'],
                 'candidate initial role fingerprints mismatch')
        run_config = write_run_config(run_config_root, family, seed, iteration, source, expected)
        candidates.append(dict(iteration=iteration, path=str(path), attempt=attempt.name,
            run_config=str(run_config), contract_source=source, contract_sha256=json_hash(expected),
            manifest_sha256=sha256(path / 'manifest.json'), joint_checkpoint_id=pair,
            num_env_steps=manifest['num_env_steps'], num_agent_steps=2 * manifest['num_env_steps'],
            label=manifest['label'], role_states=manifest['role_states'],
            initial_role_fingerprints=manifest['initial_role_fingerprints']))
    _require(len({c['manifest_sha256'] for c in candidates}) == len(CANDIDATES), 'duplicate candidate manifest')
    _require(len({c['joint_checkpoint_id'] for c in candidates}) == len(CANDIDATES), 'duplicate candidate pair id')
    _require(len({c['path'] for c in candidates}) == len(CANDIDATES), 'duplicate candidate path')
    return dict(schema=SCHEMA_RUN, variant='10m_12pair_jg_excluded', run_dir=str(run),
        family=family, seed=seed, profile_hash=result['profile_hash'], budget=contract['budget'],
        target_iteration=TARGET_ITERATION, contract=contract, contract_sha256=json_hash(contract),
        files=files, candidates=candidates,
        lineage=dict(origin_attempt=run.name, candidate_attempts=attempts,
                     heterogeneous=len(set(attempts.values())) > 1, notes=notes),
        initial_role_fingerprints=result['initial_role_fingerprints'])


def write_run_config(root, family, seed, iteration, source, contract):
    """Materialise the run_config.json evaluate.prepare_plan expects.

    The CPU trainer never wrote one; it is derived from validated archived
    contracts. Segment contracts get a distinct name so both lineages survive.
    """
    _require(root is not None, 'run_config_root required')
    directory = Path(root) / f'{family}-{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    name = 'run_config.json' if source == 'origin_contract' else f'run_config.segment_{iteration}.json'
    path = directory / name
    body = dict(contract=contract)
    if path.exists():
        _require(_read(path) == body, f'existing run_config differs: {path}')
    else:
        path.write_text(json.dumps(body, sort_keys=True, indent=2) + '\n')
    return path


def origin_receipts(*, deep=True, run_config_root=None, require_unique=True):
    """All 12 origin receipts, ordered by family then training seed.

    Reproduces admission._origin_pool's cross-family initialization check over
    the available families only (J_G is absent, so its groups are vacuous).
    """
    receipts = [origin_receipt(family, seed, deep=deep, run_config_root=run_config_root)
                for family, seed in pairs()]
    if require_unique:
        manifests = [c['manifest_sha256'] for r in receipts for c in r['candidates']]
        paths = [c['path'] for r in receipts for c in r['candidates']]
        _require(len(set(manifests)) == len(manifests), 'duplicate candidate manifest across origins')
        _require(len(set(paths)) == len(paths), 'duplicate candidate path across origins')
    by_pair = {(r['family'], r['seed']): r for r in receipts}
    for seed in SEEDS:
        for role, families in (('av_policy', ('A', 'J_L', 'J_G')),
                               ('pv_policy', ('P_L', 'J_L')),
                               ('pv_policy', ('P_G', 'J_G'))):
            present = [f for f in families if (f, seed) in by_pair]
            if len(present) < 2:
                continue
            values = {by_pair[(f, seed)]['candidates'][0]['initial_role_fingerprints'][role] for f in present}
            _require(len(values) == 1, f'cross-family {role} initialization mismatch for seed {seed}')
    return receipts


def publish_origin_receipts(root, *, deep=True):
    """Exclusively publish origins/<pair>/receipt.json; never overwrite."""
    root = Path(root)
    receipts = origin_receipts(deep=deep, run_config_root=root / 'origins')
    for receipt in receipts:
        path = root / 'origins' / f"{receipt['family']}-{receipt['seed']}" / 'receipt.json'
        _require(not path.is_symlink(), 'unsafe receipt path')
        if path.exists():
            _require(_read(path) == receipt, f'existing origin receipt differs: {path}')
        else:
            path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + '\n')
    return receipts


def validation_seed_manifest(*, runs=VALIDATION_RUNS, root=protocol.VALIDATION_ROOT):
    return protocol.seed_manifest('validation', root, density_tokens=VALIDATION_DENSITIES,
                                  fraction_tokens=VALIDATION_FRACTIONS, runs=runs)


def validation_plans(receipts):
    """20 executor plans keyed 'family:iteration'; each carries all three seeds.

    Plan JSON stays exactly the shape evaluate.prepare_plan accepts for
    purpose='validation' (frozen manifest, single B2 control, no extras).
    """
    _require(len(receipts) == len(FAMILIES) * len(SEEDS), 'validation plans require the 12 origins')
    by_pair = {(r['family'], r['seed']): r for r in receipts}
    _require(set(by_pair) == set(pairs()), 'validation plans require exactly the 12 pairs')
    manifest = [row._asdict() for row in validation_seed_manifest()]
    plans = {}
    for family in FAMILIES:
        for iteration in CANDIDATES:
            bundles = []
            for seed in SEEDS:
                origin = by_pair[(family, seed)]
                candidate = next(c for c in origin['candidates'] if c['iteration'] == iteration)
                bundles.append(dict(family=family, training_seed=seed, bundle=candidate['path'],
                                    run_config=candidate['run_config']))
            plans[validation_key(family, iteration)] = dict(purpose='validation',
                seed_manifest=manifest, physical=dict(evaluate.PHYSICAL_DEFAULTS),
                fixed_modes=list(FIXED_MODES), diagnostics=[], bundles=bundles)
    return plans


def task_groups(plan):
    """The evaluate groups (density, run) a validation plan executes."""
    from sep_research.protocol import SeedRow
    manifest = tuple(SeedRow(**row) for row in plan['seed_manifest'])
    keys = sorted({(row.canonical_density_id, row.run_index) for row in manifest})
    return keys


def _b2_identity(physical):
    return json_hash(dict(method='B2', config=physical, rule_detection=30, rule_cooldown=1))


def _score_pair(prepared, rows, baseline, candidate, plan_key, origin):
    ordered = sorted((row for row in rows
                      if (row['method'], row['training_seed']) == (origin['family'], origin['seed'])),
                     key=lambda row: row['run_index'])
    _require([row['run_index'] for row in ordered] == list(range(VALIDATION_RUNS)),
             'canonical 20 candidate runs required')
    for row, control in zip(ordered, baseline):
        protocol.validate_pair(row, control)
    speeds = [row['mean_speed_priority'] for row in ordered]
    rejected = [row['total_events']['rejected'] for row in ordered]
    b2_rejected = [row['total_events']['rejected'] for row in baseline]
    b2_mean = float(sum(b2_rejected)) / len(b2_rejected)
    rejected_mean = float(sum(rejected)) / len(rejected)
    anomaly = rejected_mean > REJECTED_FACTOR * max(b2_mean, REJECTED_FLOOR)
    return dict(family=origin['family'], seed=origin['seed'], validation_key=plan_key,
        candidate=dict(iteration=candidate['iteration'], path=candidate['path'],
                       manifest_sha256=candidate['manifest_sha256'],
                       joint_checkpoint_id=candidate['joint_checkpoint_id'],
                       num_env_steps=candidate['num_env_steps'],
                       num_agent_steps=candidate['num_agent_steps'],
                       contract_source=candidate['contract_source']),
        run_speeds=speeds, rows_sha256=json_hash(ordered),
        mean_speed_priority=absolute_summary(speeds)['mean'],
        speed_minus_B2=paired_summary(speeds, [row['mean_speed_priority'] for row in baseline],
                                      samples=10_000, seed=protocol.BOOTSTRAP_ROOT),
        rejected_mean=rejected_mean, rejected_total=sum(rejected),
        b2_rejected_mean=b2_mean, rejected_anomaly=anomaly)


def collect_scores(validation_dirs, receipts):
    """Read and verify all 20 validation outputs; return (outputs, scores, baseline)."""
    from sep_research.evaluate import prepare_plan, read_evaluation
    plans = validation_plans(receipts)
    _require(type(validation_dirs) is dict and set(validation_dirs) == set(plans),
             'exact 20 validation output keys required')
    by_pair = {(r['family'], r['seed']): r for r in receipts}
    outputs, scores, baseline, locations = {}, [], None, set()
    for key, plan in plans.items():
        path = Path(validation_dirs[key])
        _require(not path.is_symlink() and path.is_dir(), 'unsafe validation output')
        path = path.resolve()
        _require(str(path) not in locations, 'duplicate validation output path')
        locations.add(str(path))
        prepared = prepare_plan(plan)
        _require(prepared['protocol_hash'] == sha256(ROOT / 'sep_research/EXPERIMENT_PROTOCOL.md'),
                 'evaluator protocol identity mismatch')
        _require(prepared['checkpoints'][('B2', None)] == _b2_identity(plan['physical']),
                 'B2 identity mismatch')
        rows = read_evaluation(path, expected_prepared=prepared)
        b2 = sorted((row for row in rows if row['method'] == 'B2'), key=lambda row: row['run_index'])
        _require(len(rows) == 80 and len(b2) == VALIDATION_RUNS, 'complete 80-row validation invocation required')
        if baseline is None:
            baseline = b2
        _require(json_hash(baseline) == json_hash(b2), 'repeated B2 panel differs between plans')
        outputs[key] = dict(path=str(path), binding_sha256=prepared['binding'],
                            rows_sha256=json_hash(rows))
        family, iteration = key.split(':')[0], int(key.split(':')[1])
        for seed in SEEDS:
            origin = by_pair[(family, seed)]
            candidate = next(c for c in origin['candidates'] if c['iteration'] == iteration)
            scores.append(_score_pair(prepared, rows, baseline, candidate, key, origin))
    return outputs, scores, baseline


def _ordered_group(scores, pair):
    return sorted((score for score in scores if (score['family'], score['seed']) == pair),
                  key=lambda score: score['candidate']['iteration'])


def tiebreak_decision(tie, results):
    """Apply 4.5 step 5 to executed 50-run tie-break speeds.

    results maps iteration -> list of mean_speed_priority values (one per run).
    A remaining <0.05 difference resolves to the earlier checkpoint.
    """
    iterations = list(tie['iterations'])
    _require(set(results) == set(iterations), 'tie-break results must cover exactly the tied candidates')
    means = {iteration: absolute_summary(results[iteration])['mean'] for iteration in iterations}
    earlier, later = sorted(iterations)
    difference = means[later] - means[earlier]
    if abs(difference) < TIE_THRESHOLD:
        chosen, rule = earlier, 'remaining_difference_below_threshold_choose_earlier'
    else:
        chosen, rule = (later if difference > 0 else earlier), 'larger_mean_speed_priority'
    return dict(iterations=iterations, means=means, runs=TIEBREAK_RUNS,
                difference=difference, threshold=TIE_THRESHOLD, chosen=chosen, rule=rule,
                result_hashes={str(iteration): json_hash(results[iteration]) for iteration in iterations})


def pair_decision(scores, pair, tiebreak_results=None):
    """4.5 steps 1-5 for one (family, seed) pair; pure and deterministic."""
    group = _ordered_group(scores, pair)
    _require(len(group) == len(CANDIDATES), 'five candidate scores required per pair')
    flagged = [score['candidate']['iteration'] for score in group if score['rejected_anomaly']]
    admissible = [score for score in group if not score['rejected_anomaly']]
    guard_all_flagged = not admissible
    if guard_all_flagged:
        admissible = list(group)
    ranked = sorted(admissible, key=lambda score: (-score['mean_speed_priority'],
                                                   score['candidate']['iteration']))
    best = ranked[0]
    margin = None if len(ranked) < 2 else best['mean_speed_priority'] - ranked[1]['mean_speed_priority']
    tie = None
    if margin is not None and margin < TIE_THRESHOLD:
        tie = dict(family=pair[0], seed=pair[1],
                   iterations=sorted([ranked[0]['candidate']['iteration'],
                                      ranked[1]['candidate']['iteration']]),
                   margin=margin, threshold=TIE_THRESHOLD, root=TIEBREAK_ROOT, runs=TIEBREAK_RUNS,
                   status='required')
        executed = (tiebreak_results or {}).get(pair_key(*pair))
        if executed:
            decision = tiebreak_decision(tie, {int(k): v for k, v in executed['run_speeds'].items()})
            tie = dict(tie, status='resolved', decision=decision)
            best = next(score for score in group
                        if score['candidate']['iteration'] == decision['chosen'])
            status = 'selected_after_tiebreak'
        else:
            status = 'tiebreak_required'
            best = None
    else:
        status = 'selected'
    return dict(family=pair[0], seed=pair[1], status=status,
                candidate=None if best is None else best['candidate'],
                mean_speed_priority=None if best is None else best['mean_speed_priority'],
                margin=margin, tie=tie,
                excluded_rejected_anomaly=flagged, rejected_guard_all_flagged=guard_all_flagged)


def select_winners(scores, tiebreak_results=None):
    return [pair_decision(scores, pair, tiebreak_results) for pair in pairs()]


def select_pairs(validation_dirs, *, receipts, tiebreak_results=None):
    """Pure offline selection, schema sep_selection_10m_v1 (4.5)."""
    _require(len(receipts) == len(pairs()), 'selection requires the 12 origin receipts')
    outputs, scores, baseline = collect_scores(validation_dirs, receipts)
    winners = select_winners(scores, tiebreak_results)
    receipt = dict(schema=SCHEMA_SELECTION, variant='10m_12pair_jg_excluded',
        excluded_families=list(EXCLUDED_FAMILIES), families=list(FAMILIES),
        seeds=list(SEEDS), candidate_iterations=list(CANDIDATES),
        origins=receipts, validation_outputs=outputs, scores=scores,
        winners=winners, B2_rows=baseline, B2_rows_sha256=json_hash(baseline),
        ranking=dict(metric='mean_speed_priority',
            order=['reject_anomalous_candidates', 'mean_speed_priority_desc',
                   'tiebreak_50_runs_fresh_seed_below_0.05', 'earlier_iteration'],
            tiebreak_threshold=TIE_THRESHOLD, tiebreak_root=TIEBREAK_ROOT,
            tiebreak_runs=TIEBREAK_RUNS, bootstrap_samples=10_000,
            bootstrap_seed=protocol.BOOTSTRAP_ROOT, run_indices=list(range(VALIDATION_RUNS)),
            rejected_guard=dict(factor=REJECTED_FACTOR, floor=REJECTED_FLOOR,
                                metric='mean rejected role events per run')),
        tiebreak_results=tiebreak_results or None)
    json_hash(receipt)
    return receipt


def validate_selection(receipt, *, receipts, tiebreak_results=None):
    """Recompute the whole receipt from current origins and read-only outputs."""
    _require(type(receipt) is dict and receipt['schema'] == SCHEMA_SELECTION, 'selection schema')
    fresh = select_pairs({key: value['path'] for key, value in receipt['validation_outputs'].items()},
                         receipts=receipts, tiebreak_results=tiebreak_results)
    _require(json_hash(receipt) == json_hash(fresh), 'selection receipt differs from recomputation')
    return fresh


def _test_inputs(receipt):
    """Materialise the 12-pair held-out plan without executing it."""
    _require(len(receipt['winners']) == len(pairs()), 'selection must cover the 12 pairs')
    winners = {(row['family'], row['seed']): row for row in receipt['winners']}
    _require(set(winners) == set(pairs()), 'selection must cover the 12 pairs')
    pending = [pair_key(*pair) for pair, row in winners.items()
               if row['candidate'] is None or row['status'] == 'tiebreak_required']
    _require(not pending, 'test plan requires resolved winners; pending tie-break: ' + ','.join(pending))
    by_origin = {(r['family'], r['seed']): r for r in receipt['origins']}
    manifest = [row._asdict() for row in protocol.seed_manifest('test', protocol.TEST_ROOT)]
    bundles = []
    for family, seed in pairs():
        origin = by_origin[(family, seed)]
        chosen = winners[(family, seed)]['candidate']
        candidate = next(c for c in origin['candidates'] if c['iteration'] == chosen['iteration'])
        _require(candidate['manifest_sha256'] == chosen['manifest_sha256'], 'winner differs from origin candidate')
        bundles.append(dict(family=family, training_seed=seed, bundle=candidate['path'],
                            run_config=candidate['run_config']))
    return dict(schema=SCHEMA_TEST, purpose='test', variant='10m_12pair_jg_excluded',
        selection_sha256=json_hash(receipt), seed_manifest=manifest,
        physical=dict(evaluate.PHYSICAL_DEFAULTS), fixed_modes=['B1', 'B2', 'N', 'L'],
        diagnostics=[], bundles=bundles,
        matrix=[dict(family=family, seed=seed, iteration=winners[(family, seed)]['candidate']['iteration'],
                     path=winners[(family, seed)]['candidate']['path'],
                     joint_checkpoint_id=winners[(family, seed)]['candidate']['joint_checkpoint_id'])
                for family, seed in pairs()],
        planned_counts=test_plan_counts())


def build_test_plan_10m(selection_receipt_path, *, receipts, tiebreak_results=None):
    """Accept a matching 12-pair selection receipt and build the held-out plan.

    The returned document is the variant's own executable plan shape; it is
    never run by the validation stage and is not consumed by the frozen 20M
    test gate (which still requires all 15 anchored pairs). Tie-break evidence
    comes from the caller (independent on-disk distributed runs) or, failing
    that, from the receipt's own embedded copy.
    """
    path = Path(selection_receipt_path)
    receipt = _read(path)
    digest = sha256(path)
    if tiebreak_results is None:
        tiebreak_results = receipt.get('tiebreak_results')
    fresh = validate_selection(receipt, receipts=receipts, tiebreak_results=tiebreak_results)
    _require(sha256(path) == digest and json_hash(receipt) == json_hash(fresh),
             'selection changed during verification')
    return _test_inputs(receipt)


def test_plan_counts():
    """Nominal arithmetic for the held-out 12-pair test matrix (no execution).

    Derived from the actual frozen test manifest so the repetition count can
    never drift from the protocol (seed_manifest default runs=30).
    """
    manifest = protocol.seed_manifest('test', protocol.TEST_ROOT)
    runs = max(row.run_index for row in manifest) + 1
    densities = len({row.density_token for row in manifest})
    fractions = len({row.fraction_token for row in manifest})
    pairs_count = len(pairs())
    learned = densities * fractions * runs
    b1 = densities * runs
    controls = densities * fractions * runs * 3
    warmups = densities * runs
    return dict(pairs=pairs_count, runs_per_condition=runs, densities=densities, fractions=fractions,
                learned_trajectories=pairs_count * learned,
                b1_reused_trajectories=b1, control_trajectories=controls,
                physical_trajectories=pairs_count * learned + b1 + controls,
                warmup_trajectories=warmups,
                total_simulations=pairs_count * learned + b1 + controls + warmups,
                logical_rows=pairs_count * learned + (b1 * fractions) + controls,
                measurement_steps=(pairs_count * learned + b1 + controls)
                                  * evaluate.PHYSICAL_DEFAULTS['measure_steps'])
