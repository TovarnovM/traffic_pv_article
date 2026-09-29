"""Offline CPU execution and portable, fail-closed result validation."""
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import tempfile
import time

from .common import ROOT, digest, read, relative, safe, sha, tree, write, evidence_tree, copy_evidence
from .contract import BATCH, validate_plan, require_acceptance, effective_profile
from .resources import Monitor, snapshot, check_limits

COUNTERS = ('num_env_steps_sampled', 'num_agent_steps_sampled',
            'num_env_steps_trained', 'num_agent_steps_trained')


def build(row, profile, source=None):
    profile = effective_profile(profile, row['family'])
    from sep_research import train as old
    from sep_research.amendment_v1 import continuation
    args = old.build_parser().parse_args(['--family', row['family'], '--profile', 'main',
        '--seed', str(row['training_seed']), '--budget', str(row['target_budget']),
        '--out-dir', '/outputs'])
    for key in ('workers', 'minibatch_size', 'ray_num_cpus', 'num_cpus_per_env_runner',
                'num_gpus', 'object_store_memory', 'train_batch_size', 'rollout_fragment_length', 'epochs'):
        setattr(args, key, profile[key])
    config = old.build_config(args)
    contract = deepcopy(old.resolved_config(args, config))
    if source is not None:
        config, contract = continuation.derive_segment_config(source, row['start_iteration'],
                                                              target_budget=row['target_budget'])
        # Rebuilding from the row also protects family, original seed and CPU geometry.
        baseline = deepcopy(contract)
        baseline.pop('continuation')
        baseline['rllib']['seed'] = row['training_seed']
        baseline['env']['seed'] = row['training_seed']
        baseline['rllib']['env_config']['seed'] = row['training_seed']
        if baseline != old.resolved_config(args, old.build_config(args)):
            raise ValueError('source scientific/CPU contract differs from requested line')
    return config, contract


def preflight(plan, run_id, output, *, enforce_limits=True):
    validate_plan(plan)
    row = next(r for r in plan['runs'] if r['run_id'] == run_id)
    profile = effective_profile(plan['profile'], row['family'])
    acceptance = require_acceptance(plan, None)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CUDA_VISIBLE_DEVICES must be empty')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        if os.environ.get(name) != '1':
            raise ValueError('worker BLAS environment must equal one')
    from sep_research import train as old
    if old.ray.__version__ != '2.56.0' or old.ray.is_initialized() or old.torch.cuda.device_count():
        raise ValueError('unsupported Ray version, existing runtime, or visible GPU')
    config, contract = build(row, plan['profile'])
    image_manifest = ROOT / 'input_manifest.json'
    if enforce_limits and not image_manifest.exists():
        raise ValueError('production preflight requires the packaged immutable input manifest')
    if image_manifest.exists():
        inputs = read(image_manifest)
        if inputs['plan_hash'] != digest(plan) or any(sha(relative(ROOT, name)) != h for name, h in inputs['files'].items()):
            raise ValueError('immutable image input hash mismatch')
    sample = snapshot(output)
    if enforce_limits:
        check_limits(sample, profile)
    # Environment inventory is provenance; installation happens only at image build.
    from importlib.metadata import distributions
    dependency_versions = {d.metadata['Name']: d.version for d in distributions() if d.metadata['Name']}
    pre = dict(schema='cpu_preflight_v1', plan_hash=digest(plan), run_id=run_id,
                limits_enforced=enforce_limits, resources=sample, versions=contract['versions'], dependency_versions=dependency_versions,
                sources=contract['sources'], sample_timeout_s=config.sample_timeout_s)
    if plan['mode'] == 'main':
        pre['operational_authorization'] = dict(
            mode=(acceptance['mode'] if acceptance['schema'] == 'cpu_operator_release_v1'
                  else 'measured_pilot_acceptance'),
            record_schema=acceptance['schema'], record_hash=digest(acceptance), record=acceptance)
    return pre


def threads(algorithm, profile, *, configure=False):
    def visit(worker):
        import torch
        wanted = profile['learner_threads'] if worker.worker_index == 0 else 1
        if configure:
            torch.set_num_threads(wanted)
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)
        devices = {pid: str(getattr(p, 'device', 'native')) for pid, p in worker.policy_map.items()}
        if (torch.get_num_threads() != wanted or torch.get_num_interop_threads() != 1
                or any(v not in ('cpu', 'native') for v in devices.values())):
            raise ValueError('actual policy CPU/thread placement mismatch')
        return dict(worker_index=worker.worker_index, intraop=wanted, interop=1, devices=devices)
    return algorithm.env_runner_group.foreach_env_runner(visit, local_env_runner=True)


def counters(algorithm, iteration):
    values = dict(algorithm._counters)
    for k in COUNTERS:
        values.setdefault(k, 0)
    if algorithm.iteration != iteration:
        raise ValueError('iteration mismatch')
    for key in COUNTERS:
        multiplier = 2 if 'agent' in key else 1
        if type(values[key]) is not int or values[key] != multiplier * BATCH * iteration:
            raise ValueError('sampled/trained counter geometry mismatch: ' + key)
    return values


def learner_state(algorithm, contract):
    from sep_research import checkpoints
    result = checkpoints._states(algorithm, contract)
    for pid in contract['trainable_ids']:
        state = algorithm.get_policy(pid).get_state()
        optimizer = state['_optimizer_variables']
        if not optimizer or any(not o.get('state') for o in optimizer) and algorithm.iteration:
            raise ValueError('missing initialized Adam slots')
        weights = algorithm.get_policy(pid).get_weights()
        critic = {k: v for k, v in weights.items() if 'value' in k or 'critic' in k}
        if not critic:
            raise ValueError('missing critic parameters')
        result[pid].update(critic=checkpoints.fingerprint(critic),
            kl=state['current_kl_coeff'], gradient_updates=state['num_grad_updates'],
            optimizer_path=type(algorithm.get_policy(pid)._optimizers[0]).__module__ + '.' +
                           type(algorithm.get_policy(pid)._optimizers[0]).__name__)
    return result


def audit(algorithm, contract, initial, delta):
    from sep_research import train as old
    payload = old.collect_seed_audits(algorithm, contract['env']['seed_guard'], initial)
    payload.update(resumed=False, trace_complete=True, status='certified_seed_exclusion')
    old.validate_seed_audit(payload, contract['env']['seed_guard'],
        expected_workers=contract['resources']['workers'],
        create_local_env_runner=contract['rllib']['create_local_env_runner'])
    physical = payload['measurement_steps'] - initial['measurement_steps']
    # Synchronous sampler performs one physical lookahead per worker. This is
    # checked against the actual transition audit on every CPU iteration.
    lookahead = contract['resources']['workers'] if delta else 0
    if physical != delta + lookahead or algorithm.env_runner_group.num_in_flight_async_reqs():
        raise ValueError('lost batches, physical lookahead mismatch or in-flight sampling')
    payload['segment'] = dict(accepted_steps=delta, physical_steps=physical,
        observed_lookahead_steps=physical-delta,
        warmup_steps=payload['warmup_steps']-initial['warmup_steps'])
    return payload


def _source(plan, row, inputs):
    if row['source_result'] is None:
        return None, None
    root = relative(inputs, row['source_result'])
    if plan['schema'] == 'cpu_plan_v2':
        return root, _recovery_source(plan, row, root)
    result = validate_result(root, run_id=row['run_id'])
    if (digest(result) != row['source_result_hash'] or result['plan_hash'] != plan['previous_plan_hash']
            or result['endpoint']['checkpoint_id'] != row['source_checkpoint_id']
            or result['final_iteration'] != row['start_iteration']
            or result['row']['target_budget'] >= row['target_budget']
            or result['row']['segment_boundaries'] != row['segment_boundaries'][:-1]
            or result['profile_hash'] != plan['profile_hash']):
        raise ValueError('wrong terminal source, budget, schedule or profile')
    return root, result


def _recovery_source(plan, row, root):
    """Validate only the exact staged recovery/resume routes; never promote a failed result."""
    from . import interrupted, partial, recovery_partial
    from .contract import plan as make_plan, production_profile
    previous = make_plan(profile=production_profile())
    if row['source_kind'] == 'interrupted_prefix':
        manifest = interrupted.validate(root, expected_plan=previous)
        if (digest(manifest) != row['source_result_hash']
                or manifest['source_plan_hash'] != row['source_plan_hash']
                or manifest['checkpoint']['checkpoint_id'] != row['source_checkpoint_id']):
            raise ValueError('interrupted source differs from immutable plan')
        source = interrupted.learner_source(root)
        source['discarded_attempt_iterations'] = read(root / 'failed/result.json')['final_iteration'] - 150
        return source
    if row['source_kind'] == 'partial_continuation':
        manifest = partial.validate(root)
        if (digest(manifest) != row['source_result_hash']
                or manifest['source_plan_hash'] != row['source_plan_hash']
                or manifest['checkpoint']['checkpoint_id'] != row['source_checkpoint_id']
                or manifest['checkpoint']['iteration'] != row['start_iteration']
                or manifest['segment_boundaries'] != row['segment_boundaries']
                or manifest['discarded_iterations'] != row['discarded_iterations']
                or plan['previous_plan_hash'] != manifest['source_plan_hash']):
            raise ValueError('partial continuation source differs from immutable plan')
        return partial.learner_source(root)
    if row['source_kind'] == 'partial_recovery':
        manifest = recovery_partial.validate(root)
        if (digest(manifest) != row['source_result_hash']
                or manifest['source_plan_hash'] != row['source_plan_hash']
                or manifest['checkpoint']['checkpoint_id'] != row['source_checkpoint_id']
                or manifest['checkpoint']['iteration'] != row['start_iteration']
                or manifest['segment_boundaries'] != row['segment_boundaries']
                or manifest['discarded_iterations'] != row['discarded_iterations']
                or plan['previous_plan_hash'] != manifest['source_plan_hash']
                or manifest['prefix_audit_complete'] is not False
                or manifest['original_protocol_eligible'] is not False):
            raise ValueError('partial recovery source differs from immutable plan')
        return recovery_partial.learner_source(root)
    source = validate_result(root, expected_plan=previous if row['source_kind'] == 'completed_result' else None,
                             run_id=row['run_id'])
    source_plan = read(root / 'plan.json')
    if row['source_kind'] == 'completed_recovery' and (
            source_plan.get('schema') != 'cpu_plan_v2'
            or source_plan.get('purpose') != 'interrupted_recovery_5m'
            or source_plan['previous_plan_hash'] != plan['previous_plan_hash']):
        raise ValueError('source is not the completed exact 150→306 recovery')
    if (digest(source) != row['source_result_hash'] or source['plan_hash'] != row['source_plan_hash']
            or source['endpoint']['checkpoint_id'] != row['source_checkpoint_id']
            or source['final_iteration'] != row['start_iteration']
            or source['row']['target_budget'] >= row['target_budget']
            or source['row']['segment_boundaries'] != row['segment_boundaries'][:-1]
            or source['profile_hash'] != plan['profile_hash']):
        raise ValueError('wrong recovered terminal source, budget, history or profile')
    return source


def _recovery_provenance(row, source):
    from .interrupted import LIMITATION
    from .partial import LIMITATION as PARTIAL_LIMITATION
    from .recovery_partial import LIMITATION as PARTIAL_RECOVERY_LIMITATION
    partial = row['source_kind'] == 'partial_continuation'
    resumed = row['source_kind'] == 'partial_recovery'
    if row['source_kind'] in ('interrupted_prefix', 'partial_continuation', 'partial_recovery'):
        discarded = source['discarded_attempt_iterations']
    else:
        discarded = source.get('recovery_provenance', {}).get('discarded_attempt_iterations', 0)
    provenance = dict(source_kind=row['source_kind'], prefix_audit_complete=row['prefix_audit_complete'],
                original_protocol_eligible=row['original_protocol_eligible'],
                segment_boundaries=row['segment_boundaries'], discarded_attempt_iterations=discarded,
                discarded_completed_env_steps=discarded * BATCH,
                limitation=(PARTIAL_LIMITATION if partial else
                            PARTIAL_RECOVERY_LIMITATION if resumed else
                            None if row['prefix_audit_complete'] else LIMITATION))
    if partial or resumed:
        provenance['boundary_comparability'] = 'heterogeneous_authorized_resume'
        provenance['discarded_iterations'] = list(row['discarded_iterations'])
    return provenance


def train(plan, run_id, output, *, inputs=ROOT, image_digest, attempt_id,
          enforce_limits=True):
    """enforce_limits=False is restricted to in-process smoke regression tests."""
    from sep_research import train as old, checkpoints
    from sep_research.amendment_v1 import continuation
    import re
    if not re.fullmatch(r'workpc411:5000/traffic-rl-cpu@sha256:[0-9a-f]{64}', image_digest):
        raise ValueError('immutable registry image digest required')
    if not re.fullmatch(r'attempt_[0-9]{3,}', attempt_id):
        raise ValueError('invalid attempt identity')
    if not enforce_limits and plan['mode'] != 'smoke':
        raise ValueError('unconfined execution is only a diagnostic smoke test')
    output = safe(output)
    output.mkdir(parents=True, exist_ok=True)
    pre = preflight(plan, run_id, output, enforce_limits=enforce_limits)
    row = next(r for r in plan['runs'] if r['run_id'] == run_id)
    profile = effective_profile(plan['profile'], row['family'])
    source_root, source = _source(plan, row, inputs)
    if any(output.iterdir()):
        raise FileExistsError('attempt output must be empty; use a new attempt')
    # Atomic permanent writer claim. Even after SSH loss or process death it is
    # never removed; a retry must use a different attempt directory.
    write(output / 'writer.json', dict(run_id=run_id, attempt_id=attempt_id, plan_hash=digest(plan)))
    write(output / 'plan.json', plan)
    write(output / 'preflight.json', pre)
    source_hashes = evidence_tree(source_root) if source_root else None
    if source:
        # Complete portable chain, including inherited candidates and prior plans.
        copy_evidence(source_root, output / 'source')
        if evidence_tree(output / 'source') != source_hashes:
            raise ValueError('portable source copy hash mismatch')
        source_root = output / 'source'
    config, contract = build(row, plan['profile'], source['contract'] if source else None)
    metadata = dict(schema='cpu_result_v1', row=row, contract=contract, plan_hash=digest(plan),
        profile_hash=plan['profile_hash'], image_digest=image_digest, attempt_id=attempt_id,
        limits_enforced=enforce_limits, status='starting', start_iteration=row['start_iteration'],
        final_iteration=row['start_iteration'], checkpoints=[], cleanup={}, signals=[],
        source='source' if source else None, source_hashes=source_hashes,
        continuation_semantics='segmented learner continuation; deterministic environment/cache/sampling RNG reset',
        exact_uninterrupted_resume=False, numa_pinned=False,
        tool_sources={str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'sep_research/cluster_cpu').glob('*.py'))})
    if plan['schema'] == 'cpu_plan_v2':
        metadata.update(schema='cpu_result_v2', recovery_provenance=_recovery_provenance(row, source))
    if plan['mode'] == 'main':
        metadata['operational_authorization'] = pre['operational_authorization']
    algorithm = None
    owned_ray = False
    monitor = Monitor(output, profile, metadata)
    failure = None
    handlers = {}
    started = time.monotonic()
    current = row['start_iteration']
    def request_stop(signum, frame):
        metadata['signals'].append(signum)
    def save(label):
        before = time.monotonic()
        name = f'checkpoints/iteration_{current:06d}'
        manifest = checkpoints.save_bundle(algorithm, output / name, contract,
            num_env_steps=current * BATCH, iteration=current, label=label,
            initial_fingerprints=metadata['initial_role_fingerprints'])
        checkpoints.validate_bundle(output / name, contract)
        entry = dict(path=name, iteration=current, checkpoint_id=manifest['joint_checkpoint_id'],
                     manifest_hash=sha(output / name / 'manifest.json'), seconds=time.monotonic()-before)
        metadata['checkpoints'].append(entry)
        metadata['last_checkpoint'] = entry
        return entry
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, request_stop)
        monitor.start()
        os.environ.update(RAY_USAGE_STATS_ENABLED='0', RAY_CHDIR_TO_TRIAL_DIR='0', WANDB_MODE='disabled')
        old.torch.set_num_threads(1)
        if old.torch.get_num_interop_threads() != 1:
            old.torch.set_num_interop_threads(1)
        owned_ray = True
        ray_temp = tempfile.mkdtemp(prefix='cc_', dir='/tmp')
        spill = output / 'spill'
        spill.mkdir()
        ray_context = old.ray.init(address='local', _node_ip_address='127.0.0.1', include_dashboard=False,
            num_cpus=plan['profile']['ray_num_cpus'], num_gpus=0,
            object_store_memory=profile['object_store_memory'], _temp_dir=ray_temp,
            object_spilling_directory=str(spill))
        algorithm = old.FactorizedGraphPPO(config=config, storage=old.StorageContext(
            storage_path=str(output), experiment_dir_name='rllib', trial_dir_name='train'))
        if source:
            metadata['restoration'] = continuation.restore_segment_state(algorithm,
                relative(source_root, source['endpoint']['path']), source['contract'], contract)
            if learner_state(algorithm, contract) != source['final_learner_state']:
                raise ValueError('restored critic/Adam/KL/gradient state differs')
            metadata['initial_role_fingerprints'] = source['initial_role_fingerprints']
        else:
            metadata['initial_role_fingerprints'] = old.role_fingerprints(algorithm)
        metadata['threads'] = threads(algorithm, plan['profile'], configure=True)
        metadata['initial_counters'] = counters(algorithm, current)
        metadata['initial_learner_state'] = learner_state(algorithm, contract)
        initial = old.collect_seed_audits(algorithm, contract['env']['seed_guard'])
        if initial['measurement_steps']:
            raise ValueError('nonzero initial segment physical counters')
        metadata['status'] = 'running'
        with (output / 'metrics.jsonl').open('x') as stream:
            while current < row['target_iteration'] and not metadata['signals']:
                before = time.monotonic()
                result = algorithm.train()
                current += 1
                values = counters(algorithm, current)
                if old.env_steps(result) != values[COUNTERS[0]] or result['training_iteration'] != current:
                    raise ValueError('result/native counter mismatch')
                threads(algorithm, plan['profile'])
                census = audit(algorithm, contract, initial, (current-row['start_iteration']) * BATCH)
                state = learner_state(algorithm, contract)
                from ray._private.internal_api import memory_summary
                metric = dict(iteration=current, counters=values,
                    segment_deltas={k: values[k]-metadata['initial_counters'][k] for k in COUNTERS},
                    family=row['family'], training_seed=row['training_seed'], profile_hash=plan['profile_hash'],
                    seconds=time.monotonic()-before, elapsed_seconds=time.monotonic()-started,
                    learner_state=state, physical=census['segment'], result=result,
                    object_store_spill_restore=memory_summary(address=ray_context.address_info['gcs_address'], stats_only=True))
                # RLlib legitimately emits NaN for not-yet-finished episodes. The
                # complete raw metrics are preserved; correctness counters above are strict.
                stream.write(json.dumps(metric, sort_keys=True, default=old._config_json) + '\n')
                stream.flush()
                metadata['final_iteration'] = current
                if current < row['target_iteration'] and (current % 25 == 0 or current in row['candidate_iterations']):
                    save('candidate' if current in row['candidate_iterations'] else 'incomplete')
                if monitor.error:
                    raise monitor.error
            metadata['endpoint'] = (metadata['last_checkpoint'] if metadata.get('last_checkpoint', {}).get('iteration') == current
                else save('final' if current == row['target_iteration'] else 'incomplete'))
            write(output / 'seed_audit.json', audit(algorithm, contract, initial, (current-row['start_iteration'])*BATCH))
            metadata['final_counters'] = counters(algorithm, current)
            metadata['final_learner_state'] = learner_state(algorithm, contract)
        if metadata['signals']:
            raise InterruptedError('saved at an accepted iteration boundary; retry from declared segment start')
    except BaseException as exc:
        failure = exc
        if algorithm is not None:
            metadata['failed_attempt_reported_counters'] = dict(algorithm._counters)
            metadata['failed_attempt_reported_iteration'] = algorithm.iteration
            metadata['uncredited_work_policy'] = 'failed attempt adds zero steps to the accepted lineage; retry declared segment start'
            try:
                failed_audit = old.collect_seed_audits(algorithm, contract['env']['seed_guard'])
                failed_audit.update(status='failed', trace_complete=False)
                write(output/'failed_audit.json', failed_audit)
            except BaseException as audit_error:
                metadata['failed_audit_error'] = str(audit_error)
    finally:
        for name, action, active in [('algorithm_stop', lambda: algorithm.stop(), algorithm is not None),
                                     ('ray_shutdown', old.ray.shutdown, owned_ray),
                                     ('monitor_stop', monitor.stop, monitor.thread.ident is not None)]:
            try:
                if active:
                    action()
                metadata['cleanup'][name] = 'complete' if active else 'not_started'
            except BaseException as exc:
                metadata['cleanup'][name] = 'failed'
                failure = failure or exc
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    try:
        metadata['final_resources'] = snapshot(output)
        if enforce_limits:
            check_limits(metadata['final_resources'], profile)
        if source_hashes is not None and evidence_tree(source_root) != source_hashes:
            raise ValueError('source mutated during continuation')
    except BaseException as exc:
        failure = failure or exc
    if metadata['signals'] and failure is None:
        failure = InterruptedError('stop requested during finalization')
    metadata.update(status='failed' if failure else 'completed', elapsed_seconds=time.monotonic()-started)
    if failure:
        metadata.update(error_type=type(failure).__name__, error=str(failure))
    write(output / 'result.json', metadata)
    # Bind only scientific evidence; Ray logs, spill and heartbeat are auxiliary.
    files = ['plan.json', 'preflight.json', 'writer.json', 'result.json', 'metrics.jsonl', 'seed_audit.json', 'resources.jsonl']
    if failure is None:
        try:
            write(output / 'completion.json', dict(schema='cpu_completion_v1', files={f: sha(output / f) for f in files}))
            validate_result(output, expected_plan=plan, run_id=run_id)
        except BaseException:
            (output / 'completion.json').unlink(missing_ok=True)
            raise
    if failure:
        raise failure
    return metadata


def validate_result(path, *, expected_plan=None, run_id=None):
    from sep_research import checkpoints
    from sep_research.seed_control import validate_seed_audit
    path = safe(path)
    receipt = read(path / 'completion.json')
    required = {'plan.json', 'preflight.json', 'writer.json', 'result.json', 'metrics.jsonl', 'seed_audit.json', 'resources.jsonl'}
    if receipt.get('schema') != 'cpu_completion_v1' or set(receipt.get('files', {})) != required:
        raise ValueError('missing completion evidence')
    if any(sha(relative(path, name)) != h for name, h in receipt['files'].items()):
        raise ValueError('completion payload hash mismatch')
    plan = validate_plan(read(path / 'plan.json'))
    result = read(path / 'result.json')
    row = next(r for r in plan['runs'] if r['run_id'] == result['row']['run_id'])
    if (expected_plan is not None and plan != expected_plan or run_id is not None and row['run_id'] != run_id
            or result['row'] != row or result['plan_hash'] != digest(plan)
            or result['profile_hash'] != plan['profile_hash'] or result['status'] != 'completed'
            or result['signals'] or result['final_iteration'] != row['target_iteration']
            or result['cleanup'] != dict(algorithm_stop='complete', ray_shutdown='complete', monitor_stop='complete')):
        raise ValueError('identity, terminal budget, status or cleanup mismatch')
    if plan['profile']['schema'] == 'cpu_profile_v3':
        from .operator_release import verify_record
        pre = read(path / 'preflight.json')
        authorization = result.get('operational_authorization', {})
        record = verify_record(authorization.get('record', {}), plan)
        expected = dict(mode=record['mode'], record_schema=record['schema'],
                        record_hash=digest(record), record=record)
        if (authorization != expected or pre.get('operational_authorization') != expected
                or not result['limits_enforced'] or not pre['limits_enforced']):
            raise ValueError('production operational authorization or enforced limits mismatch')
    source = None
    if plan['schema'] == 'cpu_plan_v2':
        source = _recovery_source(plan, row, path / 'source')
        if (result.get('schema') != 'cpu_result_v2' or result.get('source') != 'source'
                or evidence_tree(path / 'source') != result['source_hashes']
                or result.get('recovery_provenance') != _recovery_provenance(row, source)
                or result['initial_learner_state'] != source['final_learner_state']
                or result['initial_role_fingerprints'] != source['initial_role_fingerprints']
                or result.get('exact_uninterrupted_resume') is not False):
            raise ValueError('recovery chain, restored learner or eligibility mismatch')
        restored = result.get('restoration', {})
        if (restored.get('source_checkpoint_id') != row['source_checkpoint_id']
                or restored.get('accepted_iteration') != row['start_iteration']
                or restored.get('initial_physical_measurement_steps') != 0
                or restored.get('exact_uninterrupted_resume') is not False):
            raise ValueError('missing recovery runtime restoration evidence')
    elif row['source_result']:
        source = validate_result(path / 'source', run_id=row['run_id'])
        if (evidence_tree(path / 'source') != result['source_hashes'] or digest(source) != row['source_result_hash']
                or source['endpoint']['checkpoint_id'] != row['source_checkpoint_id']
                or source['plan_hash'] != plan['previous_plan_hash']
                or source['final_iteration'] != row['start_iteration']
                or source['row']['target_budget'] >= row['target_budget']):
            raise ValueError('continuation chain mismatch')
        if result['initial_learner_state'] != source['final_learner_state']:
            raise ValueError('restoration evidence mismatch')
    _, expected_contract = build(row, plan['profile'], source['contract'] if source else None)
    if plan['mode'] == 'main' and row['start_iteration'] == 0 and row['target_budget'] != 5_000_000:
        raise ValueError('fresh main lines must start with the declared 5M budget')
    if result['start_iteration'] != row['start_iteration']:
        raise ValueError('segment start mismatch')
    inherited = set()
    # Exact v2 stages select only NEW suffix candidates; source validators own their full history.
    prior = None if plan['schema'] == 'cpu_plan_v2' else source
    prior_root = path/'source'
    while prior:
        inherited.update(e['iteration'] for e in prior['checkpoints'])
        if not prior['source']:
            break
        prior_root = prior_root/'source'
        prior = read(prior_root/'result.json')
    if not {i for i in row['candidate_iterations'] if i <= row['start_iteration']} <= inherited:
        raise ValueError('inherited candidate missing from immutable source chain')
    if result['contract'] != expected_contract:
        raise ValueError('scientific contract differs from current sources')
    entries = result['checkpoints']
    needed = {i for i in range(row['start_iteration']+1, row['target_iteration']+1)
              if i % 25 == 0 or i in row['candidate_iterations']} | {row['target_iteration']}
    if {e['iteration'] for e in entries} != needed or len(entries) != len(needed):
        raise ValueError('missing/duplicate checkpoint schedule')
    for entry in entries:
        bundle = relative(path, entry['path'])
        m = checkpoints.validate_bundle(bundle, result['contract'])
        if (sha(bundle / 'manifest.json') != entry['manifest_hash']
                or m['joint_checkpoint_id'] != entry['checkpoint_id'] or m['iteration'] != entry['iteration']
                or m['num_env_steps'] != entry['iteration'] * BATCH
                or m['initial_role_fingerprints'] != result['initial_role_fingerprints']):
            raise ValueError('checkpoint identity/counters mismatch')
    terminal = relative(path, result['endpoint']['path'])
    if read(terminal / 'manifest.json')['label'] != 'final' or native_evidence(terminal, result['contract']) != result['final_learner_state']:
        raise ValueError('terminal native learner evidence mismatch')
    if result['endpoint'] != entries[-1] or entries[-1]['iteration'] != row['target_iteration']:
        raise ValueError('source must be terminal, never a selected candidate')
    for key in COUNTERS:
        multiplier = 2 if 'agent' in key else 1
        if (result['initial_counters'][key] != multiplier * BATCH * row['start_iteration']
                or result['final_counters'][key] != multiplier * BATCH * row['target_iteration']):
            raise ValueError('cumulative counters mismatch')
    metrics = [json.loads(line) for line in (path / 'metrics.jsonl').read_text().splitlines()]
    if [m['iteration'] for m in metrics] != list(range(row['start_iteration']+1, row['target_iteration']+1)):
        raise ValueError('missing metric iterations')
    for m in metrics:
        for key in COUNTERS:
            if m['counters'][key] != (2 if 'agent' in key else 1) * BATCH * m['iteration']:
                raise ValueError('metric counters mismatch')
    if metrics[-1]['learner_state'] != result['final_learner_state']:
        raise ValueError('learner terminal evidence mismatch')
    census = read(path / 'seed_audit.json')
    contract = result['contract']
    validate_seed_audit(census, contract['env']['seed_guard'], expected_workers=plan['profile']['workers'],
                       create_local_env_runner=contract['rllib']['create_local_env_runner'])
    delta = (row['target_iteration']-row['start_iteration']) * BATCH
    if (census['measurement_steps']-census['initial']['measurement_steps'] != delta+plan['profile']['workers']
            or census['segment']['accepted_steps'] != delta):
        raise ValueError('physical/accepted geometry mismatch')
    samples = [json.loads(line) for line in (path / 'resources.jsonl').read_text().splitlines()]
    if not samples:
        raise ValueError('missing cgroup evidence')
    if result['limits_enforced']:
        check_limits(result['final_resources'], effective_profile(plan['profile'], row['family']))
    return result


def native_evidence(bundle, contract):
    """Verify trusted native payload contents after the bundle hash gate."""
    from sep_research import checkpoints
    import ray.cloudpickle as pickle
    manifest = checkpoints.validate_bundle(bundle, contract)
    bundle = Path(bundle)
    fmt = read(bundle/'algorithm/rllib_checkpoint.json')
    if (fmt['type'] != 'Algorithm' or fmt['format'] != 'cloudpickle'
            or fmt['ray_version'] != '2.56.0' or fmt['checkpoint_version'] != '1.1'):
        raise ValueError('unsupported native format')
    with (bundle/'algorithm/algorithm_state.pkl').open('rb') as stream:
        native = pickle.load(stream)
    if native['training_iteration'] != manifest['iteration']:
        raise ValueError('native iteration mismatch')
    for key in COUNTERS:
        if native['counters'].get(key, 0) != (2 if 'agent' in key else 1) * BATCH * manifest['iteration']:
            raise ValueError('native counter mismatch')
    roles = {}
    for pid in contract['policy_ids']:
        with (bundle/'algorithm/policies'/pid/'policy_state.pkl').open('rb') as stream:
            state = pickle.load(stream)
        # Policy.export_checkpoint adds format metadata absent from get_state().
        # Validate it explicitly, then compare the actual learner-state payload.
        if str(state.pop('checkpoint_version', None)) != '1.1':
            raise ValueError('unsupported native policy checkpoint version')
        weights = state['weights']
        optimizers = state.get('_optimizer_variables', [])
        entry = dict(weights=checkpoints.fingerprint(weights), optimizers=checkpoints.fingerprint(optimizers),
                     optimizer_count=len(optimizers), state=checkpoints.fingerprint({k:v for k,v in state.items()
                        if k not in ('connector_configs','policy_spec')}))
        if entry != manifest['role_states'][pid]:
            raise ValueError(f'native weights/Adam/KL/gradient fingerprint mismatch: {pid} keys={sorted(state)} differing={[k for k in entry if entry[k] != manifest["role_states"][pid][k]]}')
        if pid in contract['trainable_ids']:
            if not optimizers or any(not o.get('state') for o in optimizers):
                raise ValueError('missing initialized Adam slots')
            critic = {k:v for k,v in weights.items() if 'value' in k or 'critic' in k}
            if not critic:
                raise ValueError('missing native critic')
            entry.update(critic=checkpoints.fingerprint(critic), kl=state['current_kl_coeff'],
                         gradient_updates=state['num_grad_updates'], optimizer_path='torch.optim.adam.Adam')
        elif weights or optimizers:
            raise ValueError('native inactive role contains learned state')
        roles[pid] = entry
    return roles
