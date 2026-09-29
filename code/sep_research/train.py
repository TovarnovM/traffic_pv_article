"""Classic factorized PPO for five families; tiny is explicitly non-main.

build_parser -> build_config returns a PPOConfig. resolved_config is the strict
JSON checkpoint contract. NativePolicy actions are sampler placeholders only.
Resume restores Algorithm state, NOT environment/cache/RNG trajectories.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import platform
import tempfile
import time

from gymnasium import spaces
from ray.rllib.utils.serialization import space_to_dict

import numpy as np
import ray
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import Policy, PolicySpec
from ray.train._internal.storage import StorageContext

from snfs_traffic.rl.graph_ppo import (
    FactorizedGraphPPO, FactorizedPPOTorchPolicy, PvGraphTorchModel,
)
from sep_research.env import SepResearchEnv
from sep_research.models import PvManeuverTorchModel, PvGraphManeuverTorchModel
from sep_research.protocol import FAMILIES
from sep_research.seed_control import frozen_seed_guard, validate_seed_audit, MANIFEST_NAMES
from sep_research.checkpoints import save_bundle, restore_bundle, fingerprint

POLICY_IDS = ('pv_policy', 'av_policy')


def policy_mapping_fn(agent_id, *args, **kwargs):
    return {'pv': 'pv_policy', 'av_coordinator': 'av_policy'}[agent_id]


class SeededFactorizedPolicy(FactorizedPPOTorchPolicy):
    """Independent role seed; original PPO initialization/loss/optimizer intact."""
    def __init__(self, observation_space, action_space, config):
        torch.set_num_threads(1)
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(config['role_init_seed'])
            super().__init__(observation_space, action_space, config)


    @torch.no_grad()
    def _initialize_loss_from_dummy_batch(self, auto_remove_unneeded_view_reqs=True, stats_fn=None):
        # Discover the original views/stats without retaining an unused backward
        # graph in tower_stats on rollout workers. Real loss calls still track gradients.
        return super()._initialize_loss_from_dummy_batch(auto_remove_unneeded_view_reqs, stats_fn)

    def set_state(self, state):
        super().set_state(state)
        # Ray 2.56 exports this counter but TorchPolicyV2.set_state omits it.
        self.num_grad_updates = state['num_grad_updates']


class NativePolicy(Policy):
    def compute_actions(self, obs_batch, state_batches=None, **kwargs):
        return np.ones((len(obs_batch), *self.action_space.shape), dtype=np.int64), [], {}

    def get_weights(self):
        return {}

    def set_weights(self, weights):
        if weights:
            raise ValueError('native policy cannot accept learned weights')

    def learn_on_batch(self, samples):
        raise RuntimeError('native policy must never learn')


class TrainingCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, episode, **kwargs):
        torch.set_num_threads(1)
        episode.user_data['events'] = {}

    def on_episode_step(self, *, episode, **kwargs):
        info = episode.last_info_for('pv')
        if not info or 'role_events' not in info:
            return
        for role, events in info['role_events'].items():
            for key, value in events.items():
                name = f'{role}/{key}'
                totals = episode.user_data['events']
                totals[name] = totals.get(name, 0) + value

    def on_episode_end(self, *, episode, **kwargs):
        info = episode.last_info_for('pv')
        if not info or 'step_index' not in info:
            raise RuntimeError('missing physical episode telemetry')
        episode.custom_metrics['team_return'] = info['team_return']
        episode.custom_metrics['v_pv'] = info['episode_mean_pv_speed']
        episode.custom_metrics[f"v_pv/{info['condition_key']}"] = info['episode_mean_pv_speed']
        for name, value in episode.user_data['events'].items():
            episode.custom_metrics[name] = value / info['step_index']


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family', choices=FAMILIES, required=True)
    p.add_argument('--profile', choices=('main', 'tiny'), default='main')
    p.add_argument('--seed', type=int, default=20260914)
    for name, typ in [('workers', int), ('num-cpus-per-env-runner', float),
                      ('ray-num-cpus', int), ('num-gpus', float),
                      ('train-batch-size', int), ('minibatch-size', int),
                      ('rollout-fragment-length', int), ('epochs', int),
                      ('budget', int)]:
        p.add_argument('--' + name, type=typ)
    p.add_argument('--iterations', type=int, default=2000)
    p.add_argument('--object-store-memory', type=int, default=512 * 1024**2)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--guard-seeds', action='store_true',
                   help='enable frozen reserved-seed exclusion/audit for tiny; always enabled for main')
    p.add_argument('--allow-runtime-changes', action='store_true')
    return p


def build_config(args):
    tiny = args.profile == 'tiny'
    def setting(name, main, small):
        value = getattr(args, name)
        return (small if tiny else main) if value is None else value
    encoder = 'graph_mlp' if args.family.endswith('_G') else 'local_mlp'
    env = dict(controller_mode=args.family, pv_encoder=encoder, seed=args.seed,
               num_lanes=4, road_length=100 if tiny else 1000,
               density_values=[.1] if tiny else [.1, .2, .3, .4],
               av_fraction_values=[.95] if tiny else [.9, .95, 1.],
               warmup_steps=0 if tiny else 1000, episode_steps=8 if tiny else 1000,
               warmup_cache=True, backend='reference' if tiny else 'optimized',
               front_distance=30, back_distance=10, sensor_distance=60,
               cooldown_steps=1, vmax_hdv=5, vmax_priority=6,
               G=15, q=.99, r=.99, S=2, P1=.999, P2=.99, P3=.98, P4=.01,
               p_lane_change=.5, applied_penalty=.001, rejected_penalty=.002,
               validate=False)
    if not tiny or args.guard_seeds:
        env['seed_guard'] = frozen_seed_guard()
    ModelCatalog.register_custom_model('sep_pv_local_v1', PvManeuverTorchModel)
    ModelCatalog.register_custom_model('sep_pv_graph_v1', PvGraphManeuverTorchModel)
    ModelCatalog.register_custom_model('sep_av_graph_v1', PvGraphTorchModel)
    spaces = SepResearchEnv(env)
    active = (['av_policy'] if args.family == 'A' else ['pv_policy']
              if args.family.startswith('P_') else list(POLICY_IDS))
    policies = {}
    for agent in spaces.possible_agents:
        pid = policy_mapping_fn(agent)
        pv = agent == 'pv'
        model = dict(custom_model=('sep_pv_graph_v1' if encoder == 'graph_mlp'
                                   else 'sep_pv_local_v1') if pv else 'sep_av_graph_v1',
                     custom_model_config={} if pv else dict(hidden_dim=128, message_layers=3),
                     vf_share_layers=True)
        seed = args.seed + (1009 if pv else 2003)
        policies[pid] = PolicySpec(SeededFactorizedPolicy if pid in active else NativePolicy,
            spaces.observation_spaces[agent], spaces.action_spaces[agent],
            {'model': model, 'role_init_seed': seed})
    spaces.close()
    config = (PPOConfig(algo_class=FactorizedGraphPPO)
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(SepResearchEnv, env_config=env)
        .framework('torch')
        .env_runners(num_env_runners=setting('workers', 32, 0), num_envs_per_env_runner=1,
                     num_cpus_per_env_runner=setting('num_cpus_per_env_runner', 1, 1),
                     rollout_fragment_length=setting('rollout_fragment_length', 256, 8),
                     # Wait for every worker; a timeout drops results, not physical work.
                     sample_timeout_s=None, batch_mode='truncate_episodes')
        .resources(num_gpus=setting('num_gpus', 1, 0))
        .training(use_gae=True, use_critic=True, gamma=.995, lambda_=.95, lr=.0002, clip_param=.2,
                  entropy_coeff=.01, vf_loss_coeff=1., vf_clip_param=10.,
                  kl_coeff=.2, kl_target=.01,
                  train_batch_size=setting('train_batch_size', 16384, 16),
                  minibatch_size=setting('minibatch_size', 1024, 8),
                  num_epochs=setting('epochs', 6, 1))
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn,
                     policies_to_train=active, count_steps_by='env_steps')
        .callbacks(TrainingCallbacks)
        .debugging(seed=args.seed, log_level='ERROR'))
    config.validate()
    return config


def _config_json(value):
    if isinstance(value, spaces.Space):
        return space_to_dict(value)
    if callable(value):
        return f'{value.__module__}.{value.__qualname__}'
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'unsupported resolved configuration value: {type(value).__name__}')


def resolved_config(args, config):
    root = Path(__file__).resolve().parents[1]
    sources = [*sorted((root / 'src/snfs_traffic').rglob('*.py')),
               root / 'research/evaluate_pv_graph_ppo.py',
               *(root / 'sep_research' / name for name in
                 ('train.py', 'checkpoints.py', 'env.py', 'env_core.py', 'models.py', 'observations.py',
                  'inference.py', 'protocol.py', 'seed_control.py', 'TECHNICAL_SPEC.md')),
               *(root / 'sep_research/manifests' / name for name in MANIFEST_NAMES)]
    return dict(schema='sep_train_v1', family=args.family, profile=args.profile,
        pv_encoder=config.env_config['pv_encoder'], seed=args.seed,
        budget=args.budget if args.budget is not None else (16 if args.profile == 'tiny' else 20_000_000),
        rllib=json.loads(json.dumps(config.serialize(), default=_config_json)),
        env=config.env_config, policy_ids=list(POLICY_IDS),
        trainable_ids=list(config.policies_to_train),
        ppo={key: getattr(config, key) for key in
             ('use_gae', 'use_critic', 'gamma', 'lambda_', 'lr', 'clip_param', 'entropy_coeff', 'vf_loss_coeff',
              'vf_clip_param', 'kl_coeff', 'kl_target', 'train_batch_size',
              'minibatch_size', 'num_epochs', 'rollout_fragment_length', 'count_steps_by')},
        models={pid: spec.config for pid, spec in config.policies.items()},
        spaces={pid: {'observation': str(spec.observation_space), 'action': str(spec.action_space)}
                for pid, spec in config.policies.items()},
        resources=dict(workers=config.num_env_runners,
            num_cpus_per_env_runner=config.num_cpus_per_env_runner, num_gpus=config.num_gpus,
            ray_num_cpus=args.ray_num_cpus, object_store_memory=args.object_store_memory),
        versions=dict(ray=ray.__version__, torch=torch.__version__, python=platform.python_version(),
                      numpy=np.__version__),
        sources={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources})


def env_steps(result):
    values = [result[k] for k in ('num_env_steps_sampled_lifetime', 'num_env_steps_sampled') if k in result]
    if not values or any(type(v) is not int or v < 0 for v in values) or len(set(values)) != 1:
        raise ValueError('missing, invalid or conflicting explicit env-step counters')
    return values[0]


def role_fingerprints(algorithm):
    return {pid: fingerprint(algorithm.get_policy(pid).get_weights()) for pid in POLICY_IDS}


def _worker_seed_lifecycle(worker):
    context = ray.get_runtime_context()
    remote = worker.worker_index != 0
    return dict(worker_index=worker.worker_index,
                recreated=worker.recreated_worker,
                actor_id=context.get_actor_id() if remote else None,
                reconstructed=context.was_current_actor_reconstructed if remote else False,
                runtime=dict(pid=os.getpid(), node_id=context.get_node_id(),
                    assigned_resources=context.get_assigned_resources() if remote else {},
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    torch_threads=torch.get_num_threads(),
                    thread_environment={k: os.environ.get(k) for k in
                        ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}))


def collect_seed_audits(algorithm, guard, initial=None):
    """Certify a synchronous classic-worker segment; missing evidence fails closed.

    RuntimeContext checks actor restarts even before our initial census. The
    manager counter and incarnation/prefix comparisons cover later replacement.
    Sub-environment recovery is unsupported: a discarded precheck instance would
    have no trace accessible through foreach_env. Resume has no prior RNG/cache
    trajectory and is separately marked uncertified by the CLI.
    """
    config, group = algorithm.config, algorithm.env_runner_group
    if config.restart_failed_sub_environments or config.remote_worker_envs:
        raise ValueError('seed audit cannot prove sub-environment recovery lifecycle')
    workers = group.foreach_env_runner(_worker_seed_lifecycle, local_env_runner=True)
    nested = group.foreach_env(lambda env: env.get_seed_audit())
    after = group.foreach_env_runner(_worker_seed_lifecycle, local_env_runner=True)
    if workers != after or len(nested) != config.num_env_runners + 1:
        raise ValueError('seed audit incomplete worker retrieval/restart lifecycle')
    audits = [audit for vector in nested for audit in vector]
    payload = dict(schema='sep_seed_audit_v2', seed_fields=guard['seed_fields'],
        manifest_sha256=guard['manifest_sha256'], guard_hash=guard['fingerprint'],
        workers=workers, environments=audits, initial=initial,
        healthy_worker_ids=sorted(group.healthy_env_runner_ids()),
        remote_worker_restarts=group.num_remote_worker_restarts(),
        reset_count=sum(len(a['records']) for a in audits),
        rejected_count=sum(a['rejected_count'] for a in audits),
        measurement_steps=sum(a['measurement_steps'] for a in audits),
        warmup_steps=sum(a['warmup_steps'] for a in audits))
    validate_seed_audit(payload, guard, expected_workers=config.num_env_runners,
        envs_per_worker=config.num_envs_per_env_runner,
        create_local_env_runner=config.create_local_env_runner, require_complete=False)
    return payload


def write_seed_audit(algorithm, guard, initial, out, metadata, *, resumed=False, training_failed=False):
    """Publish a new segment atomically; never overwrite any earlier audit."""
    failure = None
    try:
        if initial is None:
            raise ValueError('seed audit missing initial lifecycle census')
        payload = collect_seed_audits(algorithm, guard, initial)
        payload['resumed'] = resumed
        payload['prior_audit_segments'] = list(metadata.get('seed_audit_segments', []))
        payload['status'] = 'uncertified_resumed' if resumed else 'certified_seed_exclusion'
        payload['trace_complete'] = not resumed and not training_failed
        if training_failed:
            payload['status'] = 'uncertified_training_failed'
        if resumed:
            payload['limitation'] = 'Algorithm-state resume does not restore environment/cache/RNG or prior actor trace'
    except Exception as exc:
        failure = exc
        payload = dict(status='failed', trace_complete=False, error=str(exc),
                       guard_hash=guard['fingerprint'])
    name = (f"seed_audit_resume_{len(metadata['resume_segments'])}.json" if resumed else 'seed_audit.json')
    target = out / name
    raw = json.dumps(payload, indent=2, sort_keys=True).encode()
    with tempfile.NamedTemporaryFile(dir=out, prefix='.seed_audit_', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(raw)
    try:
        # Atomic exclusive publication, matching the no-overwrite audit contract.
        os.link(temporary, target)
    finally:
        temporary.unlink()
    receipt = dict(file=name, sha256=hashlib.sha256(raw).hexdigest(),
                   status=payload['status'], trace_complete=payload['trace_complete'],
                   reset_count=payload.get('reset_count'), rejected_count=payload.get('rejected_count'))
    metadata.setdefault('seed_audit_segments', []).append(receipt)
    metadata['seed_audit'] = receipt
    if failure is not None:
        raise failure
    return receipt



def publish_completion_evidence(out, metadata, *, resumed=False):
    """Exclusive post-cleanup evidence, not full-main admission.

    Resume gets a separate receipt. Old receipts/hashes remain immutable and
    describe their historical files, not a subsequently appended metrics log.
    No receipt digest is written back into the metadata it binds.
    """
    if (metadata['finalization']['status'] != 'complete'
            or metadata['finalization']['algorithm_stop'] != 'complete'
            or metadata['finalization']['ray_shutdown'] != 'complete'):
        raise ValueError('completion evidence requires successful owned cleanup')
    files = ['run_config.json', 'train_metrics.jsonl']
    files.extend(receipt['file'] for receipt in metadata.get('seed_audit_segments', []))
    payload = dict(schema='sep_completion_evidence_v1', finalization='complete',
        status=metadata['status'], budget_reached=metadata['num_env_steps'] >= metadata['contract']['budget'],
        resumed=resumed, eligible_for_main_completeness=False,
        initial_counters=metadata['initial_counters'], final_counters=metadata['final_counters'],
        files={name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in files})
    name = (f"completion_evidence_resume_{len(metadata['resume_segments'])}.json"
            if resumed else 'completion_evidence.json')
    raw = json.dumps(payload, indent=2, sort_keys=True).encode()
    with tempfile.NamedTemporaryFile(dir=out, prefix='.completion_', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(raw)
    try:
        os.link(temporary, out / name)
    finally:
        temporary.unlink()


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = build_config(args)
    contract = resolved_config(args, config)
    if args.iterations < 0 or contract['budget'] < 1:
        raise ValueError('invalid iteration/budget limit')
    out = args.out_dir.resolve()
    if args.profile == 'tiny' and not (out.name.startswith('tiny_integration_') and out.parent == Path('sep_research/artifacts').resolve()):
        raise ValueError('tiny CLI output must be sep_research/artifacts/tiny_integration_*')
    if config.num_gpus > 0:
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if not visible or ',' in visible or config.num_gpus > 1:
            raise ValueError('GPU runs require exactly one explicit CUDA_VISIBLE_DEVICES index/UUID')
    if ray.is_initialized():
        raise RuntimeError('refusing existing Ray runtime')
    if args.resume:
        previous = json.loads((out / 'run_config.json').read_text())
        from sep_research.checkpoints import validate_bundle, compatible
        compatible(previous['contract'], contract, args.allow_runtime_changes)
        for receipt in previous.get('seed_audit_segments', []):
            if hashlib.sha256((out / receipt['file']).read_bytes()).hexdigest() != receipt['sha256']:
                raise ValueError('prior seed audit segment hash mismatch')
        pending = validate_bundle(args.resume, contract, allow_runtime_changes=args.allow_runtime_changes)
        rows = (out / 'train_metrics.jsonl').read_text().splitlines()
        if rows:
            last = json.loads(rows[-1])
            if (last['num_env_steps'], last['iteration']) != (pending['num_env_steps'], pending['iteration']):
                raise ValueError('resume bundle is not the metrics log tip')
    elif out.exists() and any(out.iterdir()):
        raise FileExistsError('fresh output directory is not empty')
    out.mkdir(parents=True, exist_ok=True)
    algorithm = None
    initial_audit = None
    audit_written = False
    training_failure = None
    owned_ray = False
    steps = pending['num_env_steps'] if args.resume else 0
    iteration = pending['iteration'] if args.resume else 0
    started = time.perf_counter()
    metadata = previous if args.resume else dict(contract=contract, status='created', resume_segments=[])
    if args.resume:
        metadata['resume_segments'].append(dict(bundle=str(args.resume.resolve()),
            resources=contract['resources'], resume_semantics='algorithm_state_non_bitwise'))
    metadata['eligible_for_main_completeness'] = False
    metadata['requested_iteration_cap'] = args.iterations
    metadata['initial_counters'] = dict(num_env_steps=steps, iteration=iteration)
    metadata['finalization'] = dict(status='pending')
    def write_status(status):
        metadata.update(status=status, num_env_steps=steps, iteration=iteration)
        temp = out / 'run_config.json.tmp'
        temp.write_text(json.dumps(metadata, indent=2, sort_keys=True))
        temp.replace(out / 'run_config.json')
    write_status('starting')
    try:
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
            os.environ[key] = '1'
        torch.set_num_threads(1)
        owned_ray = True
        ray.init(address='local', include_dashboard=False,
            num_cpus=args.ray_num_cpus or (2 if args.profile == 'tiny' else 34),
            num_gpus=config.num_gpus, object_store_memory=args.object_store_memory,
            _temp_dir=tempfile.mkdtemp(prefix='sr_', dir='/tmp'))
        metadata['resolved_runtime'] = dict(
            ray_cluster_resources=ray.cluster_resources(),
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            torch_threads=torch.get_num_threads(),
            torch_interop_threads=torch.get_num_interop_threads(),
            thread_environment={k: os.environ.get(k) for k in
                ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            cuda_device_count=torch.cuda.device_count(),
            cuda_devices=[dict(index=i, name=torch.cuda.get_device_properties(i).name,
                uuid=str(getattr(torch.cuda.get_device_properties(i), 'uuid', 'unavailable')))
                for i in range(torch.cuda.device_count())])
        os.environ['RAY_CHDIR_TO_TRIAL_DIR'] = '0'
        algorithm = FactorizedGraphPPO(config=config, storage=StorageContext(
            storage_path=str(out), experiment_dir_name='rllib', trial_dir_name='train'))
        metadata['resolved_runtime']['policy_devices'] = {
            pid: str(getattr(algorithm.get_policy(pid), 'device', None)) for pid in POLICY_IDS}
        if 'seed_guard' in config.env_config:
            initial_audit = collect_seed_audits(algorithm, config.env_config['seed_guard'])
        if args.resume:
            manifest = restore_bundle(algorithm, args.resume, contract,
                                      allow_runtime_changes=args.allow_runtime_changes)
            steps, iteration = manifest['num_env_steps'], manifest['iteration']
        else:
            metadata['initial_role_fingerprints'] = role_fingerprints(algorithm)
        write_status('running')
        with (out / 'train_metrics.jsonl').open('a' if args.resume else 'x') as stream:
            for _ in range(args.iterations):
                if steps >= contract['budget']:
                    break
                result = algorithm.train()
                new_steps = env_steps(result)
                if new_steps <= steps or result['training_iteration'] <= iteration:
                    raise ValueError('nonmonotonic training counters')
                steps, iteration = new_steps, result['training_iteration']
                row = dict(iteration=iteration, num_env_steps=steps,
                    num_agent_steps=result.get('num_agent_steps_sampled'),
                    family=args.family, pv_encoder=contract['pv_encoder'], profile=args.profile,
                    seed=args.seed, role_fingerprints=role_fingerprints(algorithm),
                    custom_metrics=result.get('env_runners', {}).get('custom_metrics', result.get('custom_metrics', {})),
                    elapsed_seconds=time.perf_counter()-started,
                    rss_max_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    gpu_peak_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None)
                stream.write(json.dumps(row, sort_keys=True) + '\n')
                stream.flush()
                print(json.dumps(row, sort_keys=True), flush=True)
                if iteration in (900, 1000, 1100, 1200):
                    save_bundle(algorithm, out / 'checkpoints' / f'candidate_{iteration}', contract,
                                num_env_steps=steps, iteration=iteration, label='candidate',
                                initial_fingerprints=metadata['initial_role_fingerprints'])
                elif args.profile == 'main' and iteration % 25 == 0:
                    save_bundle(algorithm, out / 'checkpoints' / f'periodic_{iteration}', contract,
                                num_env_steps=steps, iteration=iteration, label='incomplete',
                                initial_fingerprints=metadata['initial_role_fingerprints'])
            reached = steps >= contract['budget']
            label = 'tiny' if args.profile == 'tiny' else ('final' if reached else 'incomplete')
            target = out / 'checkpoints' / f'{label}_{iteration}_{steps}'
            if not (args.resume and steps == manifest['num_env_steps']):
                save_bundle(algorithm, target, contract, num_env_steps=steps, iteration=iteration,
                            label=label, initial_fingerprints=metadata['initial_role_fingerprints'])
            if 'seed_guard' in config.env_config:
                # Set before publication so a certification failure is not retried.
                audit_written = True
                write_seed_audit(algorithm, config.env_config['seed_guard'], initial_audit,
                                 out, metadata, resumed=bool(args.resume))
            write_status('finalizing')
    except BaseException as exc:
        training_failure = exc
        if algorithm is not None and 'seed_guard' in config.env_config and not audit_written:
            try:
                write_seed_audit(algorithm, config.env_config['seed_guard'], initial_audit,
                                 out, metadata, resumed=bool(args.resume), training_failed=True)
            except Exception as audit_exc:
                exc.add_note(f'seed audit failed: {audit_exc}')
        try:
            write_status('failed_or_interrupted')
        except Exception as status_exc:
            exc.add_note(f'writing failure status failed: {status_exc}')
        raise
    finally:
        cleanup_failures = []
        if algorithm is not None:
            try:
                algorithm.stop()
            except BaseException as exc:
                cleanup_failures.append(('Algorithm', exc))
        if owned_ray:
            try:
                ray.shutdown()
            except BaseException as exc:
                cleanup_failures.append(('Ray', exc))
        failure = training_failure or (cleanup_failures[0][1] if cleanup_failures else None)
        for owner, exc in cleanup_failures:
            if failure is not exc:
                failure.add_note(f'{owner} cleanup failed: {exc}')
        metadata['final_counters'] = dict(num_env_steps=steps, iteration=iteration)
        metadata['finalization'] = dict(
            status='failed' if failure else 'complete',
            algorithm_stop='failed' if any(k == 'Algorithm' for k, _ in cleanup_failures) else
                           ('complete' if algorithm is not None else 'not_created'),
            ray_shutdown='failed' if any(k == 'Ray' for k, _ in cleanup_failures) else
                         ('complete' if owned_ray else 'not_started'),
            errors=[f'{owner}: {exc}' for owner, exc in cleanup_failures])
        try:
            write_status('failed_or_interrupted' if failure else
                         ('budget_reached' if steps >= contract['budget'] else 'incomplete'))
            if failure is None:
                publish_completion_evidence(out, metadata, resumed=bool(args.resume))
        except BaseException as exc:
            if failure is not None:
                failure.add_note(f'writing finalization failed: {exc}')
            else:
                failure = exc
                metadata['finalization']['status'] = 'failed'
                try:
                    write_status('failed_or_interrupted')
                except BaseException as status_exc:
                    exc.add_note(f'writing failure status failed: {status_exc}')
        if failure is not None and training_failure is None:
            raise failure
    return 0


if __name__ == '__main__':
    # Stable class paths in checkpoints, including when invoked with python -m.
    from sep_research.train import main as canonical_main
    raise SystemExit(canonical_main())
