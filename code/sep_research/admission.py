"""Trusted-local full-main admission and selection, without Ray, pickle or Algorithm restore.

Only seal_main_profile(legacy_cache=...) publishes the designated anchor, once.
Call it in the owned devcontainer AFTER all execution code/gates are finished;
this API's existence is not readiness or launch authorization. It captures the
current runtime and derives all 15 contracts from train's config builders (no
Algorithm). load_frozen_profile() reads that anchor; validate_main_run requires
its complete returned object, reloads it, and checks current critical sources.
An arbitrary profile path, digest, or self-hashed output is not an anchor.

Receipt sep_main_validated_v1: profile_sha256 identifies the complete profile;
contract_sha256 identifies the exact template; files maps config/metrics/audit/
completion names to byte SHA256; candidates lists path, manifest_sha256, pair ID,
iteration, environment/agent steps, role_states and initial_role_fingerprints.
family, seed, run_dir and measurement_steps identify the origin. Paths locate
bytes, not identities. Selection revalidates every origin and compares caller
receipts; historical bundle eligibility flags are never rewritten. Main-run
admission alone does not select pairs or authorize heldout execution.

Selection schema sep_selection_v1 binds all 15 origins, 25 validation outputs,
75 scores, the single B2 panel and 15 whole-pair winners. validation_plans
provides the shared executor contract; select_pairs and validate_selection
rebuild trusted plans independently of output provenance, without execution.
Hashes provide integrity, NOT authenticity against local-owner fabrication.
"""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import tempfile
from importlib.metadata import version

from sep_research import protocol
from sep_research.checkpoints import POLICY_IDS, fingerprint, validate_bundle
from sep_research.export_legacy import ROOT, json_hash, sha256, load_legacy
from sep_research.seed_control import MANIFEST_NAMES, frozen_seed_guard, validate_seed_audit

PROFILE_PATH = ROOT / 'sep_research/manifests/main_profile.json'
GPU_UUIDS = ('GPU-dc2a7eac-73d3-d52b-0906-b08268eff78e',
             'GPU-227b6de9-c61e-63d4-3031-de0cea0e494f')
THREADS = dict.fromkeys(('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'), '1')
TRAIN_ARGS = ('--profile main --workers 32 --num-cpus-per-env-runner .45 '
              '--ray-num-cpus 16 --num-gpus 1 --budget 20000000 '
              '--train-batch-size 16384 --minibatch-size 1024 --epochs 6 '
              '--rollout-fragment-length 256 --object-store-memory 4294967296').split()
FINAL_ITERATION = 1221
FINAL_STEPS = 20_004_864
CANDIDATES = {f'candidate_{i}': i for i in (900, 1000, 1100, 1200)}
CANDIDATES[f'final_{FINAL_ITERATION}_{FINAL_STEPS}'] = FINAL_ITERATION


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('nonfinite JSON number: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def _read(path):
    try:
        _require(not path.is_symlink(), 'symlink input')
        return _decode(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'missing/invalid JSON: {path}') from exc


def critical_sources():
    """Union of trainer/evaluator inventories plus admission and runtime inputs.

    Top-level execution modules include future selector changes automatically;
    tests, artifacts and evolving status documents are deliberately excluded.
    """
    paths = [*sorted((ROOT / 'src/snfs_traffic').rglob('*.py')),
             *sorted((ROOT / 'sep_research').glob('*.py')),
             ROOT / 'research/evaluate_pv_graph_ppo.py',
             *(ROOT / 'sep_research' / name for name in
               ('TECHNICAL_SPEC.md', 'EXPERIMENT_PROTOCOL.md', 'RESOURCE_PROTOCOL.md',
                'devcontainer/devcontainer.json', 'devcontainer/requirements.txt')),
             *(ROOT / 'sep_research/manifests' / name for name in MANIFEST_NAMES)]
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def _publish(path, payload):
    """Atomic exclusive publication; no replacement, including dangling symlinks."""
    raw = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name + '.', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink()


def _preflight():
    # No Ray head, GPU allocation, or caller-supplied resource attestation.
    import torch
    _require(Path('/.dockerenv').exists() and ROOT == Path('/workspace'), 'seal requires owned devcontainer')
    _require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'seal preflight must be CPU-only')
    threads = {k: os.environ.get(k) for k in THREADS}
    _require(threads == THREADS and torch.get_num_threads() == 1, 'seal thread environment')
    output = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name', '--format=csv,noheader'],
                            check=True, capture_output=True, text=True, timeout=15).stdout
    devices = dict(line.strip().split(', ', 1) for line in output.splitlines() if line.strip())
    _require(all(u in devices and devices[u] == 'NVIDIA GeForce RTX 3090' for u in GPU_UUIDS),
             'approved GPUs unavailable')
    shm = os.statvfs('/dev/shm')
    shared_memory_bytes = shm.f_frsize * shm.f_blocks
    _require(shared_memory_bytes >= 20 * 1024**3, 'seal shared memory capacity')
    return dict(shared_memory_bytes=shared_memory_bytes, logical_cpu_count=os.cpu_count(),
                cpu_affinity=sorted(os.sched_getaffinity(0)), torch_threads=torch.get_num_threads(),
                torch_interop_threads=torch.get_num_interop_threads(), thread_environment=threads,
                devices={u: devices[u] for u in GPU_UUIDS},
                versions=dict(python=platform.python_version(), torch=torch.__version__,
                              numpy=version('numpy'), ray=version('ray')))


def seal_main_profile(*, legacy_cache):
    """Explicit final-readiness operation; never call during unfinished gate work.

    legacy_cache must be an existing validated published-control numeric export.
    No arbitrary config/runtime/source inputs or alternate anchor destination.
    """
    if PROFILE_PATH.exists() or PROFILE_PATH.is_symlink():
        raise FileExistsError(PROFILE_PATH)
    sources = critical_sources()
    runtime = _preflight()
    _, legacy = load_legacy(legacy_cache)
    from sep_research.train import build_parser, build_config, resolved_config
    contracts = {}
    for family in protocol.FAMILIES:
        for seed in protocol.TRAINING_SEEDS:
            args = build_parser().parse_args([*TRAIN_ARGS, '--family', family, '--seed', str(seed),
                                              '--out-dir', '/tmp/sep_main_profile_unused'])
            contracts[f'{family}:{seed}'] = resolved_config(args, build_config(args))
    from sep_research.evaluate import PHYSICAL_DEFAULTS, _BINDING_FIELDS
    physical = dict(PHYSICAL_DEFAULTS)
    for entry in legacy['models'].values():
        _require(all(entry['config'][k] == physical[k] for k in _BINDING_FIELDS),
                 'legacy control physical contract')
    fixed = {m: json_hash(legacy['models'][m]) if m.endswith('_old') else
             json_hash(dict(method=m, config=physical, rule_detection=30, rule_cooldown=1))
             for m in protocol.FIXED_CONTROLS}
    payload = dict(schema='sep_main_profile_v1', train_args=TRAIN_ARGS,
                   sources=sources, runtime=runtime, contracts=contracts,
                   seed_guard=frozen_seed_guard(), evaluation_physical=physical,
                   fixed_controls=fixed, legacy_cache=dict(path=str(Path(legacy_cache).resolve()),
                       manifest_sha256=sha256(Path(legacy_cache) / 'manifest.json'), manifest=legacy),
                   protocol_sha256=sources['sep_research/EXPERIMENT_PROTOCOL.md'])
    _require(critical_sources() == sources, 'source changed during seal')
    _publish(PROFILE_PATH, payload)
    return load_frozen_profile()


def load_frozen_profile():
    """Read designated anchor only. No writes, Ray imports, or runtime startup."""
    profile = _read(PROFILE_PATH)
    _require(profile['schema'] == 'sep_main_profile_v1', 'profile schema')
    _require(profile['sources'] == critical_sources(), 'profile critical source identity')
    _require(profile['train_args'] == TRAIN_ARGS, 'profile public training arguments')
    _require(set(profile['contracts']) == {f'{f}:{s}' for f in protocol.FAMILIES for s in protocol.TRAINING_SEEDS},
             'profile requires all 15 family/seed templates')
    _require(profile['seed_guard'] == frozen_seed_guard()
             and len(profile['seed_guard']['reserved_values']) == 5500, 'profile reserved manifest guard')
    _require(profile['protocol_sha256'] == profile['sources']['sep_research/EXPERIMENT_PROTOCOL.md'],
             'profile protocol identity')
    return profile


def _hash(value):
    _require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'invalid fingerprint')


def _roles(values, active):
    _require(type(values) is dict and set(values) == set(POLICY_IDS), 'missing role owners')
    for pid, value in values.items():
        _hash(value)
        _require(pid in active or value == fingerprint({}), 'native role fingerprint')


def _runtime(runtime, template, active, workers):
    for key in ('cpu_affinity', 'torch_threads', 'torch_interop_threads', 'thread_environment'):
        _require(json_hash(runtime[key]) == json_hash(template[key]), 'runtime ' + key)
    _require(runtime['torch_threads'] == 1 and runtime['thread_environment'] == THREADS, 'runtime threads')
    uuid = runtime['cuda_visible_devices']
    _require(uuid in GPU_UUIDS and type(runtime['cuda_device_count']) is int
             and runtime['cuda_device_count'] == 1, 'driver GPU binding')
    # Torch properties omit the nvidia-smi GPU- prefix.
    _require(runtime['cuda_devices'] == [dict(index=0, name=template['devices'][uuid], uuid=uuid[4:])],
             'driver GPU identity')
    _require(runtime['policy_devices'] == {p: 'cuda:0' if p in active else 'None' for p in POLICY_IDS},
             'policy device ownership')
    resources = runtime['ray_cluster_resources']
    _require(resources.get('CPU') == 16 and resources.get('GPU') == 1, 'Ray advertised CPU/GPU')
    _require(resources.get('object_store_memory') == 4294967296, 'Ray object store capacity')
    _require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in resources.values()),
             'invalid dynamic Ray resources')
    for worker in workers:
        remote = worker['worker_index'] != 0
        r = worker['runtime']
        _require(r['torch_threads'] == 1 and r['thread_environment'] == THREADS, 'worker threads')
        _require(r['cuda_visible_devices'] == ('' if remote else uuid), 'worker GPU visibility')
        assigned = r['assigned_resources']
        _require((assigned.get('CPU') == .45 and assigned.get('GPU', 0) == 0) if remote else assigned == {},
                 'worker assigned resources')


def _measurements(audit):
    _require(audit['initial']['measurement_steps'] == 0, 'initial measurement count')
    _require(audit['measurement_steps'] == FINAL_STEPS + 32, 'accepted plus lookahead measurement count')
    for census in (audit['initial'], audit):
        initial = census is audit['initial']
        for env in census['environments']:
            remote = env['worker_index'] != 0
            target = 0 if initial or not remote else 625153
            _require(env['measurement_steps'] == target, 'worker measurement count')
            records = env['records']
            # Initial local constructor and remote first reset each execute warmup.
            _require(records[0]['warmup_cache_hit'] is False and records[0]['warmup_steps'] == 1000,
                     'initial warm cache miss')
            _require(len(records) == (1 if initial or not remote else 627), 'main reset horizon')
            warm_sources = set()
            for i, row in enumerate(records):
                # Constructor precheck, sampler reset at zero, then 1000-step episodes.
                before = max(0, i - 1) * 1000
                _require(row['requested_warmup_steps'] == 1000
                         and row['measurement_steps_before'] == before
                         and row['previous_episode_steps'] == (1000 if i > 1 else 0), 'main reset boundaries')
                source = tuple(row['seeds'][:2])
                if row['warmup_cache_hit']:
                    _require(source in warm_sources, 'warm cache source has no prior miss')
                else:
                    warm_sources.add(source)
            # The source caches one warmed state per density. The ledger has no
            # density labels, so this checks source reuse, not reconstructed rho.
            _require(sum(not r['warmup_cache_hit'] for r in records) <= 4,
                     'warm cache exceeds main density support')
            _require(env['episode_measurement_steps'] == target % 1000, 'main episode horizon')


def validate_main_run(run_dir, *, frozen_profile):
    """Pure offline validation; return fresh membership receipt, never publish it."""
    try:
        return _validate_main_run(Path(run_dir), frozen_profile)
    except (KeyError, TypeError, AttributeError, OSError) as exc:
        raise ValueError('missing/invalid main admission evidence') from exc


def _validate_main_run(run, supplied):
    profile = load_frozen_profile()
    _require(type(supplied) is dict and json_hash(supplied) == json_hash(profile), 'unanchored frozen profile')
    _require(not run.is_symlink(), 'symlink run')
    run = run.resolve()
    config = _read(run / 'run_config.json')
    contract = config['contract']
    family, seed = contract['family'], contract['seed']
    _require(family in protocol.FAMILIES and type(seed) is int and seed in protocol.TRAINING_SEEDS,
             'main family/seed')
    expected = profile['contracts'][f'{family}:{seed}']
    _require(json_hash(contract) == json_hash(expected), 'anchored exact training contract')
    _require(contract['profile'] == 'main' and contract['budget'] == 20_000_000, 'main budget/profile')
    _require(contract['versions'] == profile['runtime']['versions'], 'anchored runtime versions')
    _require(all(profile['sources'].get(k) == v for k, v in contract['sources'].items()), 'contract source identity')
    zero = dict(iteration=0, num_env_steps=0)
    final = dict(iteration=FINAL_ITERATION, num_env_steps=FINAL_STEPS)
    _require(config['resume_segments'] == [] and not any('resume' in p.name for p in run.iterdir()), 'resumed origin')
    _require(json_hash(config['initial_counters']) == json_hash(zero)
             and json_hash(config['final_counters']) == json_hash(final)
             and config['iteration'] == FINAL_ITERATION and config['num_env_steps'] == FINAL_STEPS,
             'main initial/final counters')
    _require(config['status'] == 'budget_reached' and config['finalization'] == dict(
        status='complete', algorithm_stop='complete', ray_shutdown='complete', errors=[]), 'post-cleanup completion')
    _require(type(config['requested_iteration_cap']) is int and config['requested_iteration_cap'] >= FINAL_ITERATION,
             'main iteration cap')
    names = ('run_config.json', 'train_metrics.jsonl', 'seed_audit.json', 'completion_evidence.json')
    _require(all(not (run / name).is_symlink() for name in names), 'symlink origin evidence')
    files = {name: sha256(run / name) for name in names}
    completion = _read(run / names[-1])
    _require(json_hash(completion) == json_hash(dict(schema='sep_completion_evidence_v1', finalization='complete', status='budget_reached',
        budget_reached=True, resumed=False, eligible_for_main_completeness=False, initial_counters=zero,
        final_counters=final, files={k: v for k, v in files.items() if k != names[-1]})), 'completion evidence binding')
    audit = _read(run / 'seed_audit.json')
    validate_seed_audit(audit, profile['seed_guard'], expected_workers=32, require_complete=True)
    receipt = dict(file='seed_audit.json', sha256=files['seed_audit.json'], status='certified_seed_exclusion',
                   trace_complete=True, reset_count=audit['reset_count'], rejected_count=audit['rejected_count'])
    _require(config['seed_audit'] == receipt and config['seed_audit_segments'] == [receipt], 'audit receipt binding')
    _measurements(audit)
    active = contract['trainable_ids']
    initial = config['initial_role_fingerprints']
    _roles(initial, active)
    _runtime(config['resolved_runtime'], profile['runtime'], active, audit['workers'])
    # Strict JSON encoding rejects NaN/Infinity anywhere, including custom metrics.
    lines = (run / 'train_metrics.jsonl').read_text().splitlines()
    _require(len(lines) == FINAL_ITERATION, 'exactly 1221 metric rows required')
    rows, elapsed = [], 0
    for i, line in enumerate(lines, 1):
        row = _decode(line)
        json_hash(row)
        _require(all(type(row[k]) is int and row[k] == v for k, v in
                     dict(iteration=i, num_env_steps=16384*i, num_agent_steps=32768*i, seed=seed).items()),
                 'metric counters/seed')
        _require(all(row[k] == contract[k] for k in ('family', 'pv_encoder', 'profile')), 'metric family/profile')
        _roles(row['role_fingerprints'], active)
        if i == FINAL_ITERATION:
            for pid in active:
                _require(row['role_fingerprints'][pid] != initial[pid],
                         f'final 1221 active role {pid} weights unchanged from initialization')
        _require(type(row['elapsed_seconds']) in (int, float) and row['elapsed_seconds'] > elapsed,
                 'metric elapsed timing')
        elapsed = row['elapsed_seconds']
        _require(type(row['rss_max_kib']) is int and row['rss_max_kib'] > 0
                 and type(row['gpu_peak_bytes']) is int and row['gpu_peak_bytes'] >= 0, 'metric resource measurements')
        _require(type(row['custom_metrics']) is dict, 'custom metrics')
        rows.append(row)
    pool = run / 'checkpoints'
    _require(pool.is_dir() and not pool.is_symlink(), 'missing candidate pool')
    candidates, ids = [], set()
    _require(set(CANDIDATES) <= {p.name for p in pool.iterdir()}, 'missing candidate')
    for path in sorted(pool.iterdir()):
        name = path.name
        if name not in CANDIDATES:
            match = re.fullmatch(r'periodic_([1-9][0-9]*)', name)
            _require(match is not None, 'unexpected selectable/partial checkpoint path')
            i = int(match[1])
            _require(i < FINAL_ITERATION and i % 25 == 0 and i not in CANDIDATES.values(), 'invalid periodic checkpoint')
            label = 'incomplete'
        else:
            i = CANDIDATES[name]
            label = 'final' if i == FINAL_ITERATION else 'candidate'
        manifest = validate_bundle(path, expected)
        _require(manifest['iteration'] == i and manifest['num_env_steps'] == rows[i-1]['num_env_steps']
                 and manifest['label'] == label, 'bundle counters/label')
        pair = manifest['joint_checkpoint_id']
        _require(type(pair) is str and re.fullmatch('[0-9a-f]{32}', pair) and pair not in ids, 'bundle pair UUID')
        ids.add(pair)
        _require(manifest['initial_role_fingerprints'] == initial, 'bundle initial role fingerprints')
        _require(manifest['eligible_for_main_completeness'] is False, 'historical bundle eligibility flag')
        for pid in POLICY_IDS:
            state = manifest['role_states'][pid]
            _require(state['weights'] == rows[i-1]['role_fingerprints'][pid], 'bundle metric role fingerprint')
            for key in ('weights', 'optimizers', 'state'):
                _hash(state[key])
            _require(type(state['optimizer_count']) is int and
                     (state['optimizer_count'] > 0 if pid in active else state['optimizer_count'] == 0), 'optimizer owner')
            if pid not in active:
                _require(state['optimizers'] == fingerprint([]), 'native optimizer fingerprint')
        if name in CANDIDATES:
            candidates.append(dict(path=str(path), manifest_sha256=sha256(path / 'manifest.json'),
                joint_checkpoint_id=pair, iteration=i, num_env_steps=16384*i, num_agent_steps=32768*i,
                role_states=manifest['role_states'], initial_role_fingerprints=initial))
    _require(files == {n: sha256(run / n) for n in names}, 'origin changed during validation')
    _require(profile == load_frozen_profile(), 'profile changed during validation')
    return dict(schema='sep_main_validated_v1', profile_sha256=json_hash(profile),
                contract_sha256=json_hash(expected), run_dir=str(run), family=family, seed=seed,
                files=files, measurement_steps=audit['measurement_steps'],
                candidates=sorted(candidates, key=lambda c: c['iteration']))


def publish_main_receipt(run_dir, *, frozen_profile, path):
    """Revalidate the origin and exclusively publish; accepts no caller receipt."""
    receipt = validate_main_run(run_dir, frozen_profile=frozen_profile)
    _publish(Path(path), receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    seal = commands.add_parser('seal')
    seal.add_argument('--legacy-cache', required=True, type=Path)
    validate = commands.add_parser('validate')
    validate.add_argument('run_dir', type=Path)
    validate.add_argument('--publish', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'seal':
        result = seal_main_profile(legacy_cache=args.legacy_cache)
    elif args.publish:
        result = publish_main_receipt(args.run_dir, frozen_profile=load_frozen_profile(), path=args.publish)
    else:
        result = validate_main_run(args.run_dir, frozen_profile=load_frozen_profile())
    print(json.dumps(result, sort_keys=True, allow_nan=False))




def _selection_profile(supplied):
    # Always establish designated authority before inspecting caller locations.
    profile = load_frozen_profile()
    _require(type(supplied) is dict and json_hash(supplied) == json_hash(profile),
             'unanchored frozen profile')
    return profile


def _origin_pool(run_receipts, profile):
    _require(type(run_receipts) is list and len(run_receipts) == 15, 'exactly 15 origin receipts required')
    origins, paths, candidates = {}, set(), set()
    pair_ids, candidate_paths = set(), set()
    traces = {}
    for supplied in run_receipts:
        _require(type(supplied) is dict, 'invalid origin receipt')
        fresh = validate_main_run(supplied['run_dir'], frozen_profile=profile)
        _require(json_hash(fresh) == json_hash(supplied), 'caller origin receipt differs from revalidation')
        key = (fresh['family'], fresh['seed'])
        _require(key not in origins and fresh['run_dir'] not in paths, 'duplicate origin')
        paths.add(fresh['run_dir'])
        origins[key] = fresh
        pool = fresh['candidates']
        _require(len(pool) == 5 and {c['iteration'] for c in pool} == set(CANDIDATES.values()),
                 'exact five candidate pool required')
        for candidate in pool:
            identity = candidate['manifest_sha256']
            _require(identity not in candidates, 'duplicate candidate')
            candidates.add(identity)
            _require(candidate['joint_checkpoint_id'] not in pair_ids
                     and candidate['path'] not in candidate_paths, 'duplicate candidate identity/path')
            pair_ids.add(candidate['joint_checkpoint_id'])
            candidate_paths.add(candidate['path'])
        audit_path = Path(fresh['run_dir']) / 'seed_audit.json'
        audit = _read(audit_path)
        _require(sha256(audit_path) == fresh['files']['seed_audit.json'], 'audit changed after admission')
        # Fixed synchronous sampling and fixed episode horizons make these
        # records timing independent. Process/actor/environment UUIDs are local
        # lifecycle evidence only, already checked by validate_main_run.
        trace = []
        for census in (audit['initial'], audit):
            trace.append([{k: env[k] for k in ('worker_index', 'vector_index', 'records',
                'rejected_values', 'rejected_count', 'measurement_steps', 'warmup_steps',
                'episode_measurement_steps')} for env in sorted(census['environments'],
                    key=lambda e: (e['worker_index'], e['vector_index']))])
        digest = json_hash(trace)
        _require(traces.setdefault(key[1], digest) == digest, 'cross-family training seed execution mismatch')
    _require(set(origins) == {(f, s) for f in protocol.FAMILIES for s in protocol.TRAINING_SEEDS},
             'exact family by seed origins required')
    for seed in protocol.TRAINING_SEEDS:
        for role, families in (('av_policy', ('A', 'J_L', 'J_G')),
                               ('pv_policy', ('P_L', 'J_L')), ('pv_policy', ('P_G', 'J_G'))):
            values = {origins[(f, seed)]['candidates'][0]['initial_role_fingerprints'][role] for f in families}
            _require(len(values) == 1, f'cross-family {role} initialization mismatch')
    return [origins[(f, s)] for f in protocol.FAMILIES for s in protocol.TRAINING_SEEDS]


def _validation_plans(origins, profile):
    manifest = [r._asdict() for r in protocol.seed_manifest('validation', protocol.VALIDATION_ROOT,
        density_tokens=(30,), fraction_tokens=(100,), runs=20)]
    return {f'{family}:{iteration}': dict(purpose='validation', seed_manifest=manifest,
        physical=profile['evaluation_physical'], fixed_modes=['B2'], diagnostics=[],
        bundles=[dict(family=family, training_seed=origin['seed'], bundle=candidate['path'],
                      run_config=str(Path(origin['run_dir']) / 'run_config.json'))
                 for origin in origins if origin['family'] == family
                 for candidate in origin['candidates'] if candidate['iteration'] == iteration])
        for family in protocol.FAMILIES for iteration in sorted(CANDIDATES.values())}


def validation_plans(run_receipts, *, frozen_profile):
    """Return the 25 executor/selector plans keyed by 'family:iteration'.

    Revalidate all 15 origins and cross-family initialization/execution first.
    No output provenance is used as authority; no evaluation or writes occur.
    """
    try:
        profile = _selection_profile(frozen_profile)
        return _validation_plans(_origin_pool(run_receipts, profile), profile)
    except (KeyError, TypeError, AttributeError, OSError) as exc:
        raise ValueError('missing/invalid validation plan inputs') from exc


def _selection_rank(score):
    # CI is descriptive for every candidate but affects ordering ONLY when
    # unrounded mean speeds are exactly equal. UUIDs never break ties.
    return (-score['mean_speed_priority'], -score['speed_minus_B2']['ci95'][0],
            score['candidate']['num_env_steps'], score['candidate']['manifest_sha256'])


def select_pairs(validation_dirs, *, run_receipts, frozen_profile):
    """Pure offline selection, schema sep_selection_v1; no heldout admission.

    validation_dirs must be an exact dict 'family:iteration' -> output directory
    (the keys returned by validation_plans). run_receipts is a 15-element list;
    its order and mapping insertion order do not affect the canonical receipt.
    All 75 whole pairs compete within their own family/training seed. Repeated
    B2 rows must match in EVERY field and collapse to one 20-run panel. Bootstrap
    uses shared canonical run draws, never resamples the three training seeds.
    Hashes provide integrity, not authentication against local-owner fabrication.
    """
    try:
        profile = _selection_profile(frozen_profile)
        return _select_pairs(validation_dirs, run_receipts, profile)
    except (KeyError, TypeError, AttributeError, OSError, IndexError) as exc:
        raise ValueError('missing/invalid selection evidence') from exc


def _validation_history(path):
    # The reader owns filename/schema validation; check file safety before hashing.
    history = sorted(p for p in path.iterdir() if p.name.startswith('failed_'))
    _require(all(not p.is_symlink() and p.is_file() for p in history),
             'unsafe validation history')
    return history


def _select_pairs(validation_dirs, run_receipts, profile):
    from sep_research.evaluate import prepare_plan, read_evaluation
    from sep_research.analysis import absolute_summary, paired_summary
    origins = _origin_pool(run_receipts, profile)
    plans = _validation_plans(origins, profile)
    _require(type(validation_dirs) is dict and set(validation_dirs) == set(plans),
             'exact 25 validation output keys required')
    outputs, scores, baseline, locations = {}, [], None, set()
    for key, plan in plans.items():
        path = Path(validation_dirs[key])
        _require(not path.is_symlink(), 'symlink validation output')
        path = path.resolve()
        _require(str(path) not in locations, 'duplicate validation output path')
        locations.add(str(path))
        prepared = prepare_plan(plan)
        provenance = prepared['provenance']
        _require(provenance['runtime'] == {k: profile['runtime']['versions'][k]
                 for k in ('python', 'numpy', 'torch')}, 'anchored evaluator runtime')
        _require(all(profile['sources'].get(k) == v for k, v in provenance['sources'].items()),
                 'anchored evaluator sources')
        _require(prepared['protocol_hash'] == profile['protocol_sha256']
                 and prepared['config_hash'] == json_hash(profile['evaluation_physical'])
                 and plan['physical']['warmup_steps'] == 1000 and plan['physical']['measure_steps'] == 2000,
                 'anchored evaluation protocol/config/horizon')
        _require(prepared['checkpoints'][('B2', None)] == profile['fixed_controls']['B2'],
                 'anchored B2 identity')
        for item in plan['bundles']:
            origin = next(o for o in origins if (o['family'], o['seed']) == (item['family'], item['training_seed']))
            candidate = next(c for c in origin['candidates'] if c['path'] == item['bundle'])
            model = provenance['models'][f"{item['family']}:{item['training_seed']}"]
            _require(model['manifest_sha256'] == candidate['manifest_sha256']
                     and model['run_config_sha256'] == origin['files']['run_config.json']
                     and model['manifest']['joint_checkpoint_id'] == candidate['joint_checkpoint_id'],
                     'prepared candidate differs from admitted origin')
        history = _validation_history(path)
        output_files = [path / n for n in ('provenance.json', 'raw.json', 'raw.csv', 'counts.json')]
        output_files.extend(path / 'chunks' / f'5_{r}.json' for r in range(20))
        output_files.extend(history)
        file_hashes = {str(p.relative_to(path)): sha256(p) for p in output_files}
        rows = read_evaluation(path, expected_prepared=prepared)
        b2 = sorted((r for r in rows if r['method'] == 'B2'), key=lambda r: r['run_index'])
        _require(len(rows) == 80 and len(b2) == 20, 'complete 80-row validation invocation required')
        if baseline is None:
            baseline = b2
        _require(json_hash(baseline) == json_hash(b2), 'repeated B2 panel differs')
        _require(history == _validation_history(path)
                 and file_hashes == {str(p.relative_to(path)): sha256(p) for p in output_files},
                 'validation output changed during reading')
        outputs[key] = dict(path=str(path), binding_sha256=prepared['binding'],
                            rows_sha256=json_hash(rows), files=file_hashes)
        for item in plan['bundles']:
            origin = next(o for o in origins if (o['family'], o['seed']) == (item['family'], item['training_seed']))
            candidate = next(c for c in origin['candidates'] if c['path'] == item['bundle'])
            ordered = sorted((r for r in rows if (r['method'], r['training_seed']) ==
                             (origin['family'], origin['seed'])), key=lambda r: r['run_index'])
            _require([r['run_index'] for r in ordered] == list(range(20)), 'canonical 20 candidate runs required')
            for row, control in zip(ordered, baseline):
                protocol.validate_pair(row, control)
            speeds = [r['mean_speed_priority'] for r in ordered]
            scores.append(dict(family=origin['family'], seed=origin['seed'], candidate=candidate,
                validation_key=key, run_speeds=speeds, rows_sha256=json_hash(ordered),
                mean_speed_priority=absolute_summary(speeds)['mean'],
                speed_minus_B2=paired_summary(speeds, [r['mean_speed_priority'] for r in baseline],
                                             samples=10000, seed=protocol.BOOTSTRAP_ROOT)))
    winners = [min((s for s in scores if (s['family'], s['seed']) == (o['family'], o['seed'])),
                   key=_selection_rank) for o in origins]
    # Recheck bytes that preparation/reader bind; no simulation or second model
    # restore is needed. Main admission already validated every bundle file.
    for origin in origins:
        run = Path(origin['run_dir'])
        _require(all(sha256(run / n) == h for n, h in origin['files'].items()), 'origin changed during selection')
        for candidate in origin['candidates']:
            path = Path(candidate['path'])
            manifest = _read(path / 'manifest.json')
            _require(sha256(path / 'manifest.json') == candidate['manifest_sha256']
                     and all(sha256(path / n) == h for n, h in manifest['files'].items()),
                     'candidate changed during selection')
    for output in outputs.values():
        _require({p.name for p in _validation_history(Path(output['path']))}
                 == {n for n in output['files'] if n.startswith('failed_')},
                 'validation history changed during selection')
        _require(all(sha256(Path(output['path']) / n) == h for n, h in output['files'].items()),
                 'validation output changed during selection')
    _require(profile == load_frozen_profile(), 'profile changed during selection')
    receipt = dict(schema='sep_selection_v1', profile_sha256=json_hash(profile), origins=origins,
        validation_outputs=outputs, scores=scores, winners=winners, B2_rows=baseline,
        B2_rows_sha256=json_hash(baseline), ranking=dict(metric='mean_speed_priority',
            order=['mean_desc', 'paired_B2_ci95_lower_desc_at_exact_mean_tie', 'env_steps_asc', 'manifest_sha256_asc'],
            bootstrap_samples=10000, bootstrap_seed=protocol.BOOTSTRAP_ROOT,
            run_indices=list(range(20)), resampling='paired_runs_shared_draws',
            ci_role='descriptive_except_exact_mean_ties'))
    json_hash(receipt)  # Reject nonfinite numbers anywhere, including extras in rows.
    return receipt


def validate_selection(receipt, *, frozen_profile):
    """Recompute the entire receipt from current origins and read-only outputs.

    No claimed score, winner, self-hash, or unknown extra field grants authority.
    Future heldout callers must complete this check before reading test inputs.
    """
    try:
        profile = _selection_profile(frozen_profile)
        _require(type(receipt) is dict and receipt['schema'] == 'sep_selection_v1', 'selection schema')
        fresh = _select_pairs({k: v['path'] for k, v in receipt['validation_outputs'].items()},
                              receipt['origins'], profile)
        _require(json_hash(receipt) == json_hash(fresh), 'selection receipt differs from recomputation')
        return fresh
    except (KeyError, TypeError, AttributeError, OSError, IndexError) as exc:
        raise ValueError('missing/invalid selection receipt') from exc


if __name__ == '__main__':
    main()
