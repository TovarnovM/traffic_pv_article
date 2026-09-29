"""CPU protocol, immutable plans and cumulative budget revisions."""
from copy import deepcopy
from pathlib import Path
import re

from .common import digest, read, write

FAMILIES = ('A', 'P_L', 'P_G', 'J_L', 'J_G')
SEEDS = (20260914, 20260915, 20260916)
CALIBRATION_SEEDS = (420260914, 420260915, 420260916)
BATCH = 16384
PROFILE = dict(schema='cpu_profile_v1', status='candidate', workers=4,
    minibatch_size=256, learner_threads=8, worker_threads=1, interop_threads=1,
    ray_num_cpus=36, num_cpus_per_env_runner=1, num_gpus=0,
    object_store_memory=2 * 1024**3, memory_limit_bytes=28_000_000_000,
    shm_bytes=4 * 1024**3, cpu_quota_logical=36, disk_min_bytes=20 * 1024**3,
    resource_interval_seconds=15, checkpoint_interval=25,
    train_batch_size=BATCH, rollout_fragment_length=256, epochs=6,
    numa_affinity=False, optimizer_note='minibatch256 gives 4x updates versus1024; no gradient accumulation',
    continuation_rng='sep_amendment_segment_rng_v1')


def production_profile():
    """The single operator-authorized 16-worker production resource map."""
    profile = deepcopy(PROFILE)
    profile.update(schema='cpu_profile_v3', workers=16, family_resources={
        family: dict(memory_limit_bytes=memory, object_store_memory=store * 1024**3,
                     shm_bytes=shm * 1024**3)
        for family, memory, store, shm in (
            ('A', 28_000_000_000, 2, 4), ('P_L', 20_000_000_000, 2, 4),
            ('P_G', 36_000_000_000, 8, 12), ('J_L', 28_000_000_000, 2, 4),
            ('J_G', 48_000_000_000, 8, 12))})
    return profile


def effective_profile(profile, family):
    validate_profile(profile)
    if family not in FAMILIES:
        raise ValueError('unsupported CPU family')
    effective = deepcopy(profile)
    if profile['schema'] == 'cpu_profile_v3':
        effective.update(effective.pop('family_resources')[family])
    return effective


def iterations(budget):
    if type(budget) is not int or budget <= 0:
        raise ValueError('budget must be a positive integer')
    return (budget + BATCH - 1) // BATCH


def candidates(target):
    return sorted(set([25 * (target * k // (1221 * 25)) for k in (900, 1000, 1100, 1200)] + [target]) - {0})


def validate_profile(profile):
    if profile.get('schema') == 'cpu_profile_v3':
        if digest(profile) != digest(production_profile()):
            raise ValueError('CPU production profile mutation or unsupported settings')
        return digest(profile)
    expected = deepcopy(PROFILE)
    if profile.get('schema') == 'cpu_profile_v2':
        expected['schema'] = 'cpu_profile_v2'
        for name, allowed in [('workers', (16, 32)),
                              ('memory_limit_bytes', (28_000_000_000, 40_000_000_000, 56_000_000_000))]:
            if type(profile.get(name)) is not int or profile[name] not in allowed:
                raise ValueError('unsupported CPU retuning ' + name)
            expected[name] = profile[name]
        if (profile.get('object_store_memory') == 8 * 1024**3
                and profile.get('shm_bytes') == 12 * 1024**3
                and profile['workers'] == 16
                and profile['memory_limit_bytes'] in (40_000_000_000, 56_000_000_000)):
            expected.update(object_store_memory=8 * 1024**3, shm_bytes=12 * 1024**3)
        if digest(profile) != digest(expected):
            raise ValueError('CPU retuning profile mutation or unsupported scientific/resource settings')
        return digest(profile)
    for name, allowed in [('workers', (2, 4, 8)), ('minibatch_size', (128, 256, 512, 1024)),
                          ('learner_threads', (1, 4, 8, 16))]:
        if type(profile.get(name)) is not int or profile[name] not in allowed:
            raise ValueError('unsupported CPU candidate ' + name)
        expected[name] = profile[name]
    if profile != expected:
        raise ValueError('CPU profile mutation or unsupported scientific/resource settings')
    return digest(profile)


def plan(mode='main', budget=5_000_000, profile=None):
    profile = deepcopy(PROFILE if profile is None else profile)
    ph = validate_profile(profile)
    if mode not in ('main', 'smoke', 'pilot'):
        raise ValueError('invalid plan mode')
    target = iterations(budget)
    if profile['schema'] == 'cpu_profile_v3' and mode != 'main':
        raise ValueError('CPU production profile requires main mode')
    if profile['schema'] == 'cpu_profile_v2':
        # Main metadata is required by package.prepare; execution stays gated.
        if mode == 'pilot' or mode == 'smoke' and budget != 2 * BATCH:
            raise ValueError('CPU retuning comparison requires a two-iteration smoke')
    if mode == 'smoke' and target not in (1, 2):
        raise ValueError('smoke must contain only 1–2 full iterations')
    if mode == 'pilot' and not 10 <= target <= 20:
        raise ValueError('pilot must contain 10–20 full iterations')
    if mode == 'main' and budget < 5_000_000:
        raise ValueError('main starts at 5M')
    seeds = SEEDS if mode == 'main' else CALIBRATION_SEEDS[:1]
    from sep_research.amendment_v1.seeding import derive_segment_seed
    for seed in seeds:
        derive_segment_seed(seed, 0)
    version = profile['schema'].removeprefix('cpu_profile_')
    prefix = f'cpu-{version}-{ph[:12]}' if version in ('v2', 'v3') else 'cpu-v1'
    runs = [dict(run_id=f'{prefix}-{mode}-{family}-{seed}', family=family, training_seed=seed,
                 target_budget=budget, target_iteration=target, start_iteration=0,
                 source_checkpoint_id=None, source_result=None, source_result_hash=None,
                 cpu_profile_hash=ph, segment_boundaries=[0, target],
                 candidate_iterations=candidates(target) if mode == 'main' else [target])
            for family in FAMILIES for seed in seeds]
    return dict(schema='cpu_plan_v1', mode=mode, profile=profile, profile_hash=ph,
                previous_plan_hash=None, runs=runs)


def validate_plan(value):
    if value.get('schema') == 'cpu_plan_v2':
        return _validate_recovery_plan(value)
    if value.get('schema') != 'cpu_plan_v1' or validate_profile(value['profile']) != value['profile_hash']:
        raise ValueError('invalid plan/profile')
    rows = value['runs']
    if not rows:
        raise ValueError('empty plan')
    base = plan(value['mode'], rows[0]['target_budget'], value['profile'])
    if value['mode'] == 'main' and rows[0]['start_iteration'] == 0 and rows[0]['target_budget'] != 5_000_000:
        raise ValueError('fresh main lines must start at 5M; larger budgets require a revision')
    if len(rows) != len(base['runs']):
        raise ValueError('incomplete matrix')
    for row, expected in zip(rows, base['runs']):
        for key in ('run_id', 'family', 'training_seed', 'target_budget', 'target_iteration',
                    'cpu_profile_hash', 'candidate_iterations'):
            if row[key] != expected[key]:
                raise ValueError('matrix identity/budget/candidate mismatch: ' + key)
        boundaries = row['segment_boundaries']
        if (boundaries != sorted(set(boundaries)) or boundaries[0] != 0
                or boundaries[-1] != row['target_iteration'] or boundaries[-2] != row['start_iteration']):
            raise ValueError('invalid segment schedule')
        if row['start_iteration'] == 0:
            if any(row[k] is not None for k in ('source_checkpoint_id', 'source_result', 'source_result_hash')):
                raise ValueError('fresh line must not have a source')
        else:
            if not all(row[k] for k in ('source_checkpoint_id', 'source_result', 'source_result_hash')):
                raise ValueError('continuation source missing')
            if not re.fullmatch(r'inputs/[A-Za-z0-9_-]+', row['source_result']):
                raise ValueError('nonportable continuation source')
    if bool(value['previous_plan_hash']) != bool(rows[0]['start_iteration']):
        raise ValueError('revision lineage mismatch')
    # Shared boundaries apply to all fifteen lines.
    if len({tuple(r['segment_boundaries']) for r in rows}) != 1:
        raise ValueError('unmatched segment schedules')
    return value


def revision(previous, budget, sources):
    """sources map run IDs to already collected result directories."""
    from .runner import validate_result
    validate_plan(previous)
    new = deepcopy(previous)
    target = iterations(budget)
    for row in new['runs']:
        if budget <= row['target_budget'] or target <= row['target_iteration']:
            raise ValueError('nominal budget and rounded iteration must strictly increase')
        result = validate_result(sources[row['run_id']], expected_plan=previous, run_id=row['run_id'])
        row.update(start_iteration=row['target_iteration'], target_iteration=target, target_budget=budget,
                   source_checkpoint_id=result['endpoint']['checkpoint_id'],
                   source_result='inputs/' + row['run_id'], source_result_hash=digest(result),
                   segment_boundaries=row['segment_boundaries'][:-1] + [row['target_iteration'], target],
                   candidate_iterations=candidates(target) if new['mode'] == 'main' else [target])
    new['previous_plan_hash'] = digest(previous)
    return validate_plan(new)


def _recovery_template(purpose):
    from .interrupted import RUN_ID
    previous = plan(profile=production_profile())
    value = deepcopy(previous)
    value.update(schema='cpu_plan_v2', purpose=purpose, previous_plan_hash=digest(previous),
                 boundary_comparability='heterogeneous_authorized_recovery')
    if purpose == 'interrupted_recovery_5m':
        value['runs'] = [r for r in value['runs'] if r['run_id'] == RUN_ID]
    elif purpose not in ('recovered_matrix_10m', 'ready_cohort_10m'):
        raise ValueError('unsupported recovery purpose')
    for row in value['runs']:
        recovered = row['run_id'] == RUN_ID
        prefix = purpose == 'interrupted_recovery_5m'
        row.update(start_iteration=150 if prefix else 306,
                   target_iteration=306 if prefix else 611,
                   target_budget=5_000_000 if prefix else 10_000_000,
                   candidate_iterations=candidates(306 if prefix else 611),
                   segment_boundaries=([0, 150, 306] if prefix else
                                       [0, 150, 306, 611] if recovered else [0, 306, 611]),
                   source_kind=('interrupted_prefix' if prefix else
                                'completed_recovery' if recovered else 'completed_result'),
                   source_plan_hash=digest(previous), source_result='inputs/' + row['run_id'],
                   prefix_audit_complete=not recovered, original_protocol_eligible=not recovered)
    return value


def _recovery_resume_template():
    """Exact single-line resume of the killed J_G recovery 150->275->306; never uninterrupted."""
    from .recovery_partial import (RUN_ID, START, BOUNDARY, TARGET, DISCARDED,
                                   PREVIOUS_PLAN_HASH, CHECKPOINT_ID, PREFIX_RESULT_HASH)
    value = deepcopy(plan(profile=production_profile()))
    value.update(schema='cpu_plan_v2', purpose='interrupted_recovery_resume_5m',
                 previous_plan_hash=PREVIOUS_PLAN_HASH,
                 boundary_comparability='heterogeneous_authorized_resume',
                 resume_semantics=dict(
                     source_attempt='attempt_002', source_kind='partial_recovery',
                     saved_checkpoint='checkpoints/iteration_000275',
                     resume_only_from=('saved checkpoint 275; resume never replays the killed recovery '
                                       'attempt'),
                     discarded_iterations=list(DISCARDED),
                     discarded_note=('iterations 276..297 were computed but never checkpointed and are '
                                     'DISCARDED, not credited'),
                     rng_reset=('sep_amendment_segment_rng_v1 restarts environment/cache/episode and '
                                'Python/NumPy/Torch sampling RNG deterministically at the 275 boundary, '
                                'exactly like every existing segment resume'),
                     uninterrupted=False, bitwise=False,
                     boundary_note=('the J_G recovery history gains an extra authorized resume boundary at '
                                    '275, so boundary_comparability is heterogeneous; iterations 151..158 '
                                    'of the failed original attempt and 276..297 of the killed recovery '
                                    'are never credited, and prefix_audit_complete/original_protocol_'
                                    'eligible stay false')))
    value['runs'] = [r for r in value['runs'] if r['run_id'] == RUN_ID]
    row = value['runs'][0]
    row.update(start_iteration=BOUNDARY, target_iteration=TARGET, target_budget=5_000_000,
               candidate_iterations=[i for i in candidates(TARGET) if i > BOUNDARY],
               segment_boundaries=[0, START, BOUNDARY, TARGET],
               source_kind='partial_recovery', source_plan_hash=PREVIOUS_PLAN_HASH,
               source_result='inputs/' + RUN_ID, source_checkpoint_id=CHECKPOINT_ID,
               prefix_audit_complete=False, original_protocol_eligible=False,
               discarded_iterations=list(DISCARDED),
               discarded_completed_env_steps=len(DISCARDED) * BATCH)
    value['runs'][0]['source_result_hash'] = PREFIX_RESULT_HASH  # provisional; constructor binds
    return value


def _partial_continuation_template():
    """Exact single-line resume of the killed P_G14 continuation; never uninterrupted."""
    from .partial import (RUN_ID, BOUNDARY, TARGET, DISCARDED,
                          PREVIOUS_PLAN_HASH, CHECKPOINT_ID, COMPLETED_RESULT_HASH)
    value = deepcopy(plan(profile=production_profile()))
    value.update(schema='cpu_plan_v2', purpose='interrupted_continuation_10m',
                 previous_plan_hash=PREVIOUS_PLAN_HASH,
                 boundary_comparability='heterogeneous_authorized_resume',
                 resume_semantics=dict(
                     source_attempt='attempt_002', source_kind='partial_continuation',
                     saved_checkpoint='checkpoints/iteration_000575',
                     resume_only_from='saved checkpoint 575; resume never replays the killed attempt',
                     discarded_iterations=list(DISCARDED),
                     discarded_note=('iterations 576..586 were computed but never checkpointed and are '
                                     'DISCARDED, not credited'),
                     rng_reset=('sep_amendment_segment_rng_v1 restarts environment/cache/episode and '
                                'Python/NumPy/Torch sampling RNG deterministically at the 575 boundary, '
                                'exactly like every existing segment resume'),
                     uninterrupted=False, bitwise=False,
                     boundary_note=('the line history gains a second authorized resume boundary at 575, so '
                                    'boundary_comparability is heterogeneous; earlier campaign candidates '
                                    '450/500/550 stay owned by the preserved attempt_002 source and are not '
                                    're-declared by this plan')))
    value['runs'] = [r for r in value['runs'] if r['run_id'] == RUN_ID]
    row = value['runs'][0]
    row.update(start_iteration=BOUNDARY, target_iteration=TARGET, target_budget=10_000_000,
               candidate_iterations=[i for i in candidates(TARGET) if i > BOUNDARY],
               segment_boundaries=[0, 306, BOUNDARY, TARGET],
               source_kind='partial_continuation', source_plan_hash=PREVIOUS_PLAN_HASH,
               source_result='inputs/' + RUN_ID, source_checkpoint_id=CHECKPOINT_ID,
               prefix_audit_complete=True, original_protocol_eligible=True,
               discarded_iterations=list(DISCARDED),
               discarded_completed_env_steps=len(DISCARDED) * BATCH)
    value['runs'][0]['source_result_hash'] = COMPLETED_RESULT_HASH  # provisional; constructor binds
    return value


def _validate_partial_continuation_plan(value):
    from .partial import RUN_ID, PREVIOUS_PLAN_HASH, CHECKPOINT_ID
    expected = _partial_continuation_template()
    if [r.get('run_id') for r in value.get('runs', [])] != [RUN_ID]:
        raise ValueError('partial continuation plan must contain exactly the one line')
    row = value['runs'][0]
    if (not isinstance(row.get('source_result_hash'), str)
            or not re.fullmatch(r'[0-9a-f]{64}', row['source_result_hash'])
            or row.get('source_plan_hash') != PREVIOUS_PLAN_HASH
            or row.get('source_checkpoint_id') != CHECKPOINT_ID):
        raise ValueError('invalid partial continuation source binding')
    expected['runs'][0]['source_result_hash'] = row['source_result_hash']
    if digest(value) != digest(expected):
        raise ValueError('unsupported partial continuation plan, history or source mutation')
    return value


def _validate_recovery_resume_plan(value):
    from .recovery_partial import RUN_ID, PREVIOUS_PLAN_HASH, CHECKPOINT_ID
    expected = _recovery_resume_template()
    if [r.get('run_id') for r in value.get('runs', [])] != [RUN_ID]:
        raise ValueError('recovery resume plan must contain exactly the one line')
    row = value['runs'][0]
    if (not isinstance(row.get('source_result_hash'), str)
            or not re.fullmatch(r'[0-9a-f]{64}', row['source_result_hash'])
            or row.get('source_plan_hash') != PREVIOUS_PLAN_HASH
            or row.get('source_checkpoint_id') != CHECKPOINT_ID):
        raise ValueError('invalid recovery resume source binding')
    expected['runs'][0]['source_result_hash'] = row['source_result_hash']
    if digest(value) != digest(expected):
        raise ValueError('unsupported recovery resume plan, history or source mutation')
    return value


def _validate_recovery_plan(value):
    if value.get('purpose') == 'interrupted_recovery_resume_5m':
        return _validate_recovery_resume_plan(value)
    if value.get('purpose') == 'interrupted_continuation_10m':
        return _validate_partial_continuation_plan(value)
    expected = _recovery_template(value.get('purpose'))
    if value.get('purpose') == 'ready_cohort_10m':
        selected = {r['run_id'] for r in value['runs']}
        expected['runs'] = [r for r in expected['runs'] if r['run_id'] in selected]
        if not expected['runs']:
            raise ValueError('empty or unknown ready cohort')
    if len(value['runs']) != len(expected['runs']):
        raise ValueError('incomplete exact recovery matrix')
    for row, wanted in zip(value['runs'], expected['runs']):
        for name in ('source_result_hash', 'source_plan_hash'):
            if not isinstance(row.get(name), str) or not re.fullmatch(r'[0-9a-f]{64}', row[name]):
                raise ValueError('invalid recovery source hash')
        if not isinstance(row.get('source_checkpoint_id'), str) or not re.fullmatch(
                r'[0-9a-f]{32}', row['source_checkpoint_id']):
            raise ValueError('invalid recovery checkpoint ID')
        if row['source_kind'] != 'completed_recovery' and row['source_plan_hash'] != wanted['source_plan_hash']:
            raise ValueError('wrong original source plan hash')
        for key in ('source_result_hash', 'source_checkpoint_id', 'source_plan_hash'):
            wanted[key] = row[key]
    if digest(value) != digest(expected):
        raise ValueError('unsupported recovery plan, history or eligibility mutation')
    return value


def recovery_plan(previous, interrupted_path):
    """Exact authorized single-line recovery 150→306; not a completed 5M prefix."""
    from .interrupted import validate
    if previous != plan(profile=production_profile()):
        raise ValueError('recovery requires canonical production 5M campaign')
    manifest = validate(interrupted_path, expected_plan=previous)
    value = _recovery_template('interrupted_recovery_5m')
    value['runs'][0].update(source_checkpoint_id=manifest['checkpoint']['checkpoint_id'],
                            source_result_hash=digest(manifest))
    return validate_plan(value)


def recovered_revision(previous, sources):
    """Exact 14 original completed306 + one recovery-completed306 → total611."""
    if previous != plan(profile=production_profile()):
        raise ValueError('recovered revision requires canonical production 5M campaign')
    value = _recovery_template('recovered_matrix_10m')
    if set(sources) != {r['run_id'] for r in previous['runs']}:
        raise ValueError('recovered revision requires all fifteen source identities')
    return _completed_revision(previous, value, sources)


def cohort_revision(previous, sources_subset):
    """Continue only ready canonical production lines from completed306 to total611."""
    if digest(previous) != digest(plan(profile=production_profile())):
        raise ValueError('cohort revision requires canonical production 5M campaign')
    identities = {r['run_id'] for r in previous['runs']}
    if not sources_subset or not set(sources_subset) <= identities:
        raise ValueError('ready cohort requires a nonempty subset of original fifteen identities')
    value = _recovery_template('ready_cohort_10m')
    value['runs'] = [r for r in value['runs'] if r['run_id'] in sources_subset]
    return _completed_revision(previous, value, sources_subset)


def partial_continuation_plan(previous, partial_path):
    """Exact authorized single-line resume 575→611 of the killed P_G14 continuation."""
    from .partial import validate, PREVIOUS_PLAN_HASH
    validate_plan(previous)
    if digest(previous) != PREVIOUS_PLAN_HASH:
        raise ValueError('partial continuation requires the exact ready cohort campaign revision')
    manifest = validate(partial_path, expected_plan=previous)
    value = _partial_continuation_template()
    value['runs'][0]['source_result_hash'] = digest(manifest)
    return validate_plan(value)


def recovery_resume_plan(previous, partial_path):
    """Exact authorized single-line resume 275→306 of the killed J_G recovery."""
    from .recovery_partial import validate, PREVIOUS_PLAN_HASH
    validate_plan(previous)
    if digest(previous) != PREVIOUS_PLAN_HASH:
        raise ValueError('recovery resume requires the exact interrupted recovery plan')
    manifest = validate(partial_path, expected_plan=previous)
    value = _recovery_resume_template()
    value['runs'][0]['source_result_hash'] = digest(manifest)
    return validate_plan(value)


def _completed_revision(previous, value, sources):
    from .runner import validate_result
    from .interrupted import RUN_ID
    from .recovery_partial import PREVIOUS_PLAN_HASH as RECOVERY_RESUME_SOURCE_PLAN
    for row in value['runs']:
        result = validate_result(sources[row['run_id']],
                                 expected_plan=None if row['run_id'] == RUN_ID else previous,
                                 run_id=row['run_id'])
        source_plan = read(Path(sources[row['run_id']]) / 'plan.json')
        if row['run_id'] == RUN_ID:
            # The J_G source is either the exact completed 150→306 recovery or its
            # authorized 275→306 resume; both keep the line permanently ineligible.
            purpose = source_plan.get('purpose') if source_plan.get('schema') == 'cpu_plan_v2' else None
            if purpose == 'interrupted_recovery_5m':
                anchor = digest(previous)
            elif purpose == 'interrupted_recovery_resume_5m':
                anchor = RECOVERY_RESUME_SOURCE_PLAN
            else:
                anchor = None
            if anchor is None or source_plan.get('previous_plan_hash') != anchor:
                raise ValueError('J_G source must be the exact completed recovery lineage')
        row.update(source_checkpoint_id=result['endpoint']['checkpoint_id'],
                   source_result_hash=digest(result), source_plan_hash=digest(source_plan))
    return validate_plan(value)


def require_acceptance(plan_value, path):
    """Require measured acceptance or an explicit release for this exact plan."""
    if plan_value['mode'] == 'main':
        if plan_value['profile']['schema'] == 'cpu_profile_v2':
            raise ValueError('main launch blocked: CPU retuning requires future production acceptance')
        from .common import ROOT
        from .acceptance import verify
        try:
            from .operator_release import DEFAULT_PATH, SCHEMA, verify as verify_release
            if path is None and DEFAULT_PATH.exists():
                try:
                    return verify_release(DEFAULT_PATH, plan_value)
                except (OSError, KeyError, ValueError, TypeError):
                    # A release for another plan never substitutes for measured evidence.
                    pass
            target = path or ROOT / 'sep_research/cluster_cpu/acceptance/acceptance.json'
            if path is not None and read(target).get('schema') == SCHEMA:
                return verify_release(target, plan_value)
            if plan_value['profile']['schema'] == 'cpu_profile_v3':
                raise ValueError('production v3 requires its exact operator release')
            return verify(target, plan_value['profile'])
        except (OSError, KeyError, ValueError, TypeError) as exc:
            raise ValueError('main launch blocked: CPU candidate has no verified production acceptance') from exc


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('main', 'smoke', 'pilot'), default='main')
    parser.add_argument('--budget', type=int, default=5_000_000)
    parser.add_argument('--output', required=True)
    parser.add_argument('--profile', help='candidate profile JSON; only permitted CPU knobs may change')
    parser.add_argument('--previous')
    parser.add_argument('--sources', help='JSON map of run_id to collected result directory')
    args = parser.parse_args(argv)
    value = (revision(read(args.previous), args.budget, read(args.sources)) if args.previous
             else plan(args.mode, args.budget, read(args.profile) if args.profile else None))
    write(args.output, validate_plan(value))


if __name__ == '__main__':
    main()
