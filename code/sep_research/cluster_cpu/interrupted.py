"""Exact, explicitly uncertified J_G checkpoint-150 prefix; never a completed result."""
from copy import deepcopy
import shutil

from .common import digest, read, write, safe, relative, sha, tree

SCHEMA = 'cpu_interrupted_prefix_v1'
RUN_ID = 'cpu-v3-9539aded8e94-main-J_G-20260916'
BOUNDARY = 150
CHECKPOINT = 'checkpoints/iteration_000150'
LIMITATION = ('Missing complete prefix seed/lifecycle census; checkpoint learner state is verified, '
              'but the failed prefix is not a certified completed segment. Environment/cache and '
              'sampling RNG restart at iteration 150; original protocol eligibility is not restored.')
FILES = ('plan.json', 'preflight.json', 'writer.json', 'result.json', 'metrics.jsonl', 'resources.jsonl')


def _expected(root):
    from .contract import plan, production_profile, BATCH
    from .runner import build, native_evidence, COUNTERS
    from .operator_release import verify_record
    from sep_research import checkpoints
    import json

    previous = read(root / 'failed/plan.json')
    if previous != plan(profile=production_profile()):
        raise ValueError('interrupted source requires the canonical fresh production 5M plan')
    result = read(root / 'failed/result.json')
    row = next(r for r in previous['runs'] if r['run_id'] == RUN_ID)
    if (result.get('schema') != 'cpu_result_v1' or result['row'] != row or result['plan_hash'] != digest(previous)
            or result['profile_hash'] != previous['profile_hash'] or result['status'] != 'failed'
            or result['source'] is not None or result['start_iteration'] != 0
            or not BOUNDARY <= result['final_iteration'] < row['target_iteration']
            or not result['limits_enforced']):
        raise ValueError('wrong interrupted source identity, status or budget')
    pre = read(root / 'failed/preflight.json')
    record = verify_record(result['operational_authorization']['record'], previous)
    auth = dict(mode=record['mode'], record_schema=record['schema'], record_hash=digest(record), record=record)
    if (result['operational_authorization'] != auth or pre['operational_authorization'] != auth
            or not pre['limits_enforced'] or pre['plan_hash'] != digest(previous)
            or pre['run_id'] != RUN_ID):
        raise ValueError('interrupted prefix authorization mismatch')
    writer = read(root / 'failed/writer.json')
    if writer != dict(run_id=RUN_ID, attempt_id=result['attempt_id'], plan_hash=digest(previous)):
        raise ValueError('interrupted writer identity mismatch')
    if result['attempt_id'] != 'attempt_001' or result['image_digest'] != (
            'workpc411:5000/traffic-rl-cpu@sha256:'
            '5d5c7ebb19e4e16eab9a6e01d4347b61831ba6b5c751089bd440a04b506a62f2'):
        raise ValueError('unsupported interrupted attempt/image')
    _, contract = build(row, previous['profile'])
    if result['contract'] != contract:
        raise ValueError('interrupted scientific contract mismatch')
    entries = [e for e in result['checkpoints'] if e['iteration'] == BOUNDARY]
    if len(entries) != 1 or entries[0]['path'] != CHECKPOINT:
        raise ValueError('missing or ambiguous checkpoint 150')
    entry = entries[0]
    bundle = relative(root, CHECKPOINT)
    manifest = checkpoints.validate_bundle(bundle, contract)
    if (manifest['iteration'] != BOUNDARY or manifest['num_env_steps'] != BOUNDARY * BATCH
            or manifest['label'] != 'incomplete'
            or manifest['joint_checkpoint_id'] != entry['checkpoint_id']
            or sha(bundle / 'manifest.json') != entry['manifest_hash']
            or manifest['initial_role_fingerprints'] != result['initial_role_fingerprints']):
        raise ValueError('interrupted checkpoint identity/counters mismatch')
    metrics = [json.loads(line) for line in (root / 'failed/metrics.jsonl').read_text().splitlines()]
    prefix = [m for m in metrics if m['iteration'] <= BOUNDARY]
    if ([m['iteration'] for m in metrics] != list(range(1, result['final_iteration'] + 1))
            or [m['iteration'] for m in prefix] != list(range(1, BOUNDARY + 1))):
        raise ValueError('missing interrupted prefix metric iterations')
    for metric in metrics:
        if (metric['family'] != row['family'] or metric['training_seed'] != row['training_seed']
                or metric['profile_hash'] != previous['profile_hash']):
            raise ValueError('interrupted metric identity mismatch')
        for key in COUNTERS:
            if metric['counters'][key] != metric['iteration'] * BATCH * (2 if 'agent' in key else 1):
                raise ValueError('interrupted metric counters mismatch')
    if native_evidence(bundle, contract) != prefix[-1]['learner_state']:
        raise ValueError('checkpoint 150 native learner state differs from metric')
    # These bytes are observations, not a replacement for a missing complete prefix census.
    if not (root / 'failed/resources.jsonl').read_text().strip():
        raise ValueError('missing interrupted resource evidence')
    files = {f'failed/{name}': sha(root / 'failed' / name) for name in FILES}
    if (root / 'failed/failed_audit.json').exists():
        files['failed/failed_audit.json'] = sha(root / 'failed/failed_audit.json')
    files.update({CHECKPOINT + '/' + name: value for name, value in tree(bundle).items()})
    return dict(schema=SCHEMA, run_id=RUN_ID, source_plan_hash=digest(previous),
                source_result_hash=digest(result), checkpoint=deepcopy(entry), files=files,
                prefix_audit_complete=False, original_protocol_eligible=False, limitation=LIMITATION)


def validate(path, *, expected_plan=None):
    root = safe(path)
    manifest = read(root / 'interrupted.json')
    if manifest.get('schema') != SCHEMA:
        raise ValueError('unsupported interrupted prefix schema')
    # Verify the allowlist and bytes BEFORE loading trusted native checkpoint pickle.
    names = set(manifest.get('files', {}))
    mandatory = {f'failed/{name}' for name in FILES}
    if not mandatory <= names or any(name not in mandatory | {'failed/failed_audit.json'}
                                    and not name.startswith(CHECKPOINT + '/') for name in names):
        raise ValueError('invalid interrupted evidence allowlist')
    if any(sha(relative(root, name)) != value for name, value in manifest['files'].items()):
        raise ValueError('interrupted evidence hash mismatch')
    if set(tree(root)) != names | {'interrupted.json'}:
        raise ValueError('unexpected interrupted evidence files')
    if expected_plan is not None and read(root / 'failed/plan.json') != expected_plan:
        raise ValueError('interrupted source plan mismatch')
    if manifest != _expected(root):
        raise ValueError('interrupted provenance or eligibility mismatch')
    return manifest


def prepare(source, checkpoint, destination, *, expected_plan):
    """Publish a NEW portable evidence directory; never edit the failed source."""
    source, checkpoint, destination = map(safe, (source, checkpoint, destination))
    if read(source / 'plan.json') != expected_plan:
        raise ValueError('interrupted source plan mismatch')
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'failed').mkdir()
    for name in FILES + ('failed_audit.json',):
        if name == 'failed_audit.json' and not (source / name).exists():
            continue
        shutil.copyfile(safe(source / name), destination / 'failed' / name)
    target = destination / CHECKPOINT
    target.mkdir(parents=True)
    for name in tree(checkpoint):
        output = relative(target, name)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(relative(checkpoint, name), output)
    manifest = _expected(destination)
    write(destination / 'interrupted.json', manifest)
    return validate(destination, expected_plan=expected_plan)


def evidence_tree(path):
    root = safe(path)
    manifest = validate(root)
    return dict(manifest['files'], **{'interrupted.json': sha(root / 'interrupted.json')})


def copy_evidence(source, destination):
    files = evidence_tree(source)
    destination = safe(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for name in files:
        target = relative(destination, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(relative(source, name), target)
    if evidence_tree(destination) != files:
        raise ValueError('interrupted portable copy mismatch')
    return files


def learner_source(path):
    """Internal restore descriptor, explicitly NOT a cpu_result/completed prefix."""
    root = safe(path)
    manifest = validate(root)
    result = read(root / 'failed/result.json')
    from .runner import native_evidence
    return dict(source_kind='interrupted_prefix', contract=result['contract'],
                endpoint=manifest['checkpoint'], initial_role_fingerprints=result['initial_role_fingerprints'],
                final_learner_state=native_evidence(relative(root, CHECKPOINT), result['contract']))
