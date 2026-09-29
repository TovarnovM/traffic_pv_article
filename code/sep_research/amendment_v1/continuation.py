"""Segmented Algorithm-state continuation, never exact/uninterrupted resume.

Stable public API: derive_segment_seed(base_training_seed, start_iteration),
derive_segment_config(source_contract, start_iteration, *, target_budget), and
restore_segment_state(algorithm, source_bundle, source_contract, destination_contract).
The config helper returns (PPOConfig, destination contract). Build a NEW Algorithm
with that config and private storage before restoring; exclusively own it until
this function returns. Migration changes only budget, rllib.seed, env.seed,
rllib.env_config.seed, and the declared continuation metadata. Original role
initialization and every other contract field remain fixed. Environments/cache/episodes and Python/NumPy/Torch sampling
RNG restart at the shared boundary. Learner weights, critics, Adam, KL, gradient
updates, counters and iteration continue. Source contracts remain immutable.

Only trusted-local bundles produced by save_bundle are accepted. Hash validation
is integrity, not authentication: native pickle can execute code. We read ONLY
fixed native checkpoint filenames after validation, never caller-selected pickle
paths or external model formats. This has the same trust boundary as restore_bundle.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from sep_research import checkpoints, protocol
from sep_research.seed_control import frozen_seed_guard
from sep_research.amendment_v1.seeding import (DOMAIN, MAX_WORKERS, _integer,
    _forbidden, _safe_seed, derive_segment_seed)
from sep_research.train import (build_parser, build_config, resolved_config,
    collect_seed_audits, role_fingerprints, _config_json)

def _builder_args(contract):
    if contract['family'] not in protocol.FAMILIES or contract['profile'] not in ('main', 'tiny'):
        raise ValueError('invalid family/profile')
    args = build_parser().parse_args(['--family', contract['family'],
        '--profile', contract['profile'], '--out-dir', '/tmp/amendment_unused'])
    args.seed = contract['seed']
    args.budget = contract['budget']
    args.guard_seeds = True
    for key, value in contract['resources'].items():
        setattr(args, key, value)
    for key, dest in (('train_batch_size', 'train_batch_size'),
                      ('minibatch_size', 'minibatch_size'),
                      ('num_epochs', 'epochs'),
                      ('rollout_fragment_length', 'rollout_fragment_length')):
        setattr(args, dest, contract['ppo'][key])
    return args


def _rebuild(contract):
    """Reconstruct both bootstrap and segment contracts using the frozen builder."""
    args = _builder_args(contract)
    config = build_config(args)  # Original role initialization seeds FIRST.
    plan = contract.get('continuation')
    if plan is not None:
        if (set(plan) != {'schema', 'base_training_seed', 'start_iteration', 'segment_seed'}
                or plan['schema'] != DOMAIN or plan['base_training_seed'] != args.seed
                or plan['segment_seed'] != derive_segment_seed(args.seed, plan['start_iteration'])):
            raise ValueError('invalid declared segment RNG plan')
        config.debugging(seed=plan['segment_seed'])
        config.environment(env_config={**config.env_config, 'seed': plan['segment_seed']})
    expected = deepcopy(resolved_config(args, config))
    if plan is not None:
        expected['continuation'] = deepcopy(plan)
    if expected != contract:
        raise ValueError('contract differs from frozen builder/source/runtime')
    if (config.num_env_runners > MAX_WORKERS or config.num_envs_per_env_runner != 1
            or config.num_env_runners < 0):
        raise ValueError('unsupported seed geometry')
    derive_segment_seed(args.seed, 0)  # Validate bootstrap base/effective seeds too.
    return args, config


def derive_segment_config(source_contract, start_iteration, *, target_budget):
    """Return config and strict destination contract; only RNG/budget may migrate.

    budget is the logical target (5M initially, 6M/10M later), never inherited
    from a bootstrap's 20M execution metadata. No old manifest is edited.
    start_iteration is the shared boundary, e.g. 100, 306, then 367.
    """
    args, _ = _rebuild(source_contract)
    start = _integer(start_iteration, 'start_iteration')
    if start <= source_contract.get('continuation', {}).get('start_iteration', -1):
        raise ValueError('segment boundary must advance')
    if _integer(target_budget, 'target_budget') == 0:
        raise ValueError('target_budget must be positive')
    args.budget = target_budget
    config = build_config(args)
    seed = derive_segment_seed(args.seed, start)
    config.debugging(seed=seed)
    config.environment(env_config={**config.env_config, 'seed': seed})
    contract = deepcopy(resolved_config(args, config))
    contract['continuation'] = dict(schema=DOMAIN, base_training_seed=args.seed,
                                   start_iteration=start, segment_seed=seed)
    return config, contract


def validate_segment_migration(source_contract, destination_contract, iteration):
    """Validate the complete migration before native restore; no runtime bypass."""
    _rebuild(destination_contract)
    _, expected = derive_segment_config(source_contract, iteration,
                                         target_budget=destination_contract['budget'])
    if expected != destination_contract:
        raise ValueError('unapproved segment migration')


def _runtime_rllib(config):
    value = json.loads(json.dumps(config.serialize(), default=_config_json))
    # Ray's constructor adds this serialization tag; it is not a training option.
    if 'class' in value:
        if value.pop('class') != 'ray.rllib.algorithms.ppo.ppo.PPOConfig':
            raise ValueError('unexpected runtime config class')
    return value


def _expected_worker_rllib(contract, worker_index):
    expected = dict(contract['rllib'])
    # EnvRunnerGroup._setup merges these only for the local learner, even on Torch.
    if worker_index == 0:
        expected['tf_session_args'] = {**expected['tf_session_args'],
                                       **expected['local_tf_session_args']}
    return expected


def _worker_state(worker):
    from ray.rllib.evaluation.sampler import SyncSampler
    if worker.last_batch is not None or (worker.sampler is not None
                                        and not isinstance(worker.sampler, SyncSampler)):
        raise ValueError('destination sampler already used or asynchronous')
    return dict(worker_index=worker.worker_index, worker_seed=worker.seed,
        config_seed=worker.config.seed, env_config=deepcopy(worker.config.env_config),
        rllib=_runtime_rllib(worker.config),
        roles={pid: deepcopy(policy.get_state()) for pid, policy in worker.policy_map.items()})


def restore_segment_state(algorithm, source_bundle, source_contract, destination_contract):
    """Restore and verify a fresh, exclusively owned Algorithm; return audit evidence.

    On any failure discard/stop this destination: no rollback or sampling is safe.
    Physical initial counters are zero while accepted Algorithm counters continue.
    Effective policy_spec configs retain source seed metadata after native restore;
    worker/global/environment seeds come from the destination, not those specs.
    """
    manifest = checkpoints.validate_bundle(source_bundle, source_contract)
    validate_segment_migration(source_contract, destination_contract, manifest['iteration'])
    if destination_contract['budget'] <= manifest['num_env_steps']:
        raise ValueError('logical target must exceed accepted source steps')
    args, rebuilt = _rebuild(destination_contract)
    if type(algorithm) is not rebuilt.algo_class or type(algorithm.config) is not type(rebuilt):
        raise ValueError('unexpected Algorithm/config class')
    from ray.rllib.policy.policy import PolicySpec
    actual_config = algorithm.config.copy(copy_frozen=False)
    actual_config.policies = {pid: PolicySpec(*spec) if isinstance(spec, tuple) else spec
                              for pid, spec in actual_config.policies.items()}
    actual = deepcopy(resolved_config(args, actual_config))
    actual['rllib'] = _runtime_rllib(actual_config)
    actual['continuation'] = deepcopy(destination_contract['continuation'])
    if actual != destination_contract:
        raise ValueError('actual destination Algorithm configuration mismatch')
    if algorithm.iteration or any(algorithm._counters.values()):
        raise ValueError('destination Algorithm must be fresh')
    group = algorithm.env_runner_group
    if group.num_in_flight_async_reqs():
        raise ValueError('in-flight destination work')
    before = collect_seed_audits(algorithm, destination_contract['env']['seed_guard'])
    if before['measurement_steps'] or any(a['pending_transitions'] for a in before['environments']):
        raise ValueError('destination physical environment already sampled')
    fresh_workers = group.foreach_env_runner(_worker_state, local_env_runner=True)
    for worker in fresh_workers:
        expected = _expected_worker_rllib(destination_contract, worker['worker_index'])
        changed = sorted(k for k in expected.keys() | worker['rllib'].keys()
                         if expected.get(k) != worker['rllib'].get(k))
        if changed:
            raise ValueError(f'actual worker {worker["worker_index"]} runtime mismatch: {changed}')

    # Fixed native format only; validate_bundle already checked every file hash.
    import ray.cloudpickle as pickle
    root = Path(source_bundle).resolve() / 'algorithm'
    metadata = json.loads((root / 'rllib_checkpoint.json').read_text())
    if (metadata['type'] != 'Algorithm' or metadata['format'] != 'cloudpickle'
            or metadata['ray_version'] != '2.56.0' or metadata['checkpoint_version'] != '1.1'):
        raise ValueError('unsupported native checkpoint format')
    with (root / 'algorithm_state.pkl').open('rb') as stream:
        native = pickle.load(stream)
    policies = {}
    for pid in checkpoints.POLICY_IDS:
        with (root / 'policies' / pid / 'policy_state.pkl').open('rb') as stream:
            policies[pid] = pickle.load(stream)
    algorithm.restore(str(root))
    if (checkpoints._states(algorithm, destination_contract) != manifest['role_states']
            or role_fingerprints(algorithm) != {p: s['weights'] for p, s in manifest['role_states'].items()}
            or dict(algorithm._counters) != dict(native['counters'])
            or algorithm.iteration != native['training_iteration']
            or algorithm.iteration != manifest['iteration']
            or algorithm._counters['num_env_steps_sampled'] != manifest['num_env_steps']):
        raise ValueError('restored learner/optimizer/counters mismatch')
    for pid, saved in policies.items():
        restored = algorithm.get_policy(pid).get_state()
        for key in ('current_kl_coeff', 'num_grad_updates', 'policy_spec'):
            if checkpoints.fingerprint(restored.get(key)) != checkpoints.fingerprint(saved.get(key)):
                raise ValueError(f'restored {pid} {key} mismatch')

    # Ray's local KL setter pops current_kl_coeff before its native remote broadcast.
    # Re-broadcast a NEW state via the supported worker API, then verify all roles.
    worker_state = algorithm.env_runner.get_state()
    group.foreach_env_runner(lambda w: w.set_state(deepcopy(worker_state)), local_env_runner=False)
    workers = group.foreach_env_runner(_worker_state, local_env_runner=True)
    seed = destination_contract['continuation']['segment_seed']
    effective = []
    for worker in workers:
        if (worker['rllib'] != _expected_worker_rllib(destination_contract, worker['worker_index'])
                or worker['config_seed'] != seed or worker['worker_seed'] != seed + worker['worker_index']
                or worker['env_config'] != destination_contract['env']):
            raise ValueError('worker seed/config out of sync')
        for pid, state in worker['roles'].items():
            local = algorithm.get_policy(pid).get_state()
            for key in set(local) | set(state):
                if key != 'connector_configs' and checkpoints.fingerprint(state.get(key)) != checkpoints.fingerprint(local.get(key)):
                    raise ValueError(f'worker {worker["worker_index"]} {pid} {key} out of sync')
        effective.append(dict(worker_index=worker['worker_index'], global_seed=worker['worker_seed'],
            environment_seed=seed + 1001 * worker['worker_index'],
            policy_spec_configs={pid: state['policy_spec']['config'] for pid, state in worker['roles'].items()}))
    after = collect_seed_audits(algorithm, destination_contract['env']['seed_guard'], before)
    if after['measurement_steps'] or group.num_in_flight_async_reqs():
        raise ValueError('sampling occurred during restore')
    # Explicit deterministic restart after restore, independent of pickle/build work.
    def restart_rng(worker):
        from ray.rllib.utils.debug import update_global_seed_if_necessary
        update_global_seed_if_necessary('torch', seed + worker.worker_index)
    group.foreach_env_runner(restart_rng, local_env_runner=True)
    return json.loads(json.dumps(dict(
        semantics='segmented_algorithm_state_new_environment_and_sampling_rng',
        exact_uninterrupted_resume=False, source_checkpoint_id=manifest['joint_checkpoint_id'],
        accepted_iteration=algorithm.iteration, accepted_counters=dict(algorithm._counters),
        initial_physical_measurement_steps=after['measurement_steps'], initial_audit=after,
        effective_workers=effective, destination_contract=destination_contract), default=_config_json))
