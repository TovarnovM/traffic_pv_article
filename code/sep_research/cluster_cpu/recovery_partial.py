"""Exact, explicitly non-terminal J_G checkpoint-275 prefix of a killed recovery.

attempt_002 continued the failed original J_G 5M line from checkpoint 150 under the
authorised interrupted_recovery_5m plan and reached iteration 297, but the host lost
power after checkpoint 275 was written. Iterations 276..297 have no saved bundle and
are discarded; this module never presents the prefix as a completed result, never
certifies an uninterrupted or bitwise resume, and records the outage/clock facts that
ended the attempt. The line ancestry stays prefix_audit_complete=false and
original_protocol_eligible=false, exactly as the 150->306 recovery left it.
"""
from copy import deepcopy
import shutil

from .common import copy_evidence, digest, read, write, relative, safe, sha, tree

SCHEMA = 'cpu_partial_recovery_v1'
RUN_ID = 'cpu-v3-9539aded8e94-main-J_G-20260916'
FAMILY = 'J_G'
TRAINING_SEED = 20260916
START = 150
BOUNDARY = 275
FINAL = 297
TARGET = 306
DISCARDED = tuple(range(BOUNDARY + 1, FINAL + 1))
CHECKPOINT = 'checkpoints/iteration_000275'
CHECKPOINT_ID = '022c35b4ca3b4bec87a4820c271306cd'
CHECKPOINT_MANIFEST_HASH = 'c8b7d01d04cfefc5da64f7bb981e8f2fe9296e6de32cb6b27a2e08fb0588b80e'
CHECKPOINT_LABEL = 'candidate'  # iteration275 is a selected candidate of the 150->306 recovery plan
PREVIOUS_PLAN_HASH = 'f1666335d2fa26aad790f217d8be3debdbb027bd1635f91c6ae64ba50badb828'
RECOVERY_PURPOSE = 'interrupted_recovery_5m'
PREVIOUS_ATTEMPT = 'attempt_002'
IMAGE_DIGEST = ('workpc411:5000/traffic-rl-cpu@sha256:'
                '06dbb2c2e7f0a180dbd83835b1e6accbe7335ee9182555063af49719435fd077')
SOURCE_CONTAINER_ID = '40f328e3cc79e47eaf46d07ff8c1ff688d7cd3f6c6dbf2386b7374a85abe34f3'
SOURCE_DEPLOYMENT_HASH = '8c5ec68f678fa306ba27a4557a663303ed4c07e63abf58127cad497e9aed4cfb'
CAMPAIGN_PLAN_HASH = '64d6e655ae044994f2e313cf42ed2063adb3167773baaf46fd0ffe109fe3ef59'
PREFIX_CHECKPOINT_ID = '4958723433fb4bc8997e13a33c5bdb4f'
PREFIX_RESULT_HASH = 'f0ec30a52376485d994636716c8e0f0709133a585c2e41c2c71e40ca5b69a288'
RESOURCES_SHA256 = 'f1d8c4ff65f2f90034e20043a39bcdc3656185b8e58b2a94676fbe6616d96ad2'
RESOURCES_BYTES = 58720256
RESOURCES_LINES = 14037
FILES = ('plan.json', 'preflight.json', 'writer.json', 'metrics.jsonl', 'heartbeat.json',
         'failed_audit.json', 'outage.json')
LIMITATION = ('Killed recovery attempt_002 prefix: learner/critic/Adam/KL/gradient state is verified only '
              'at checkpoint 275, iterations 276..297 were computed but never checkpointed and are '
              'discarded, and environment/cache/episode/sampling RNG restarts deterministically at 275 '
              'under sep_amendment_segment_rng_v1. The J_G line stays prefix_audit_complete=false and '
              'original_protocol_eligible=false; this is not an uninterrupted or bitwise continuation.')


def _validate_outage(outage, heartbeat):
    state = outage.get('host_state', {})
    if (outage.get('schema') != 'cpu_partial_recovery_outage_v1' or outage.get('run_id') != RUN_ID
            or outage.get('attempt_id') != PREVIOUS_ATTEMPT
            or outage.get('container_id') != SOURCE_CONTAINER_ID
            or outage.get('image_digest') != IMAGE_DIGEST
            or outage.get('deployment_hash') != SOURCE_DEPLOYMENT_HASH
            or outage.get('restart_count') != 0 or outage.get('oom') is not False
            or state.get('Status') != 'exited' or state.get('ExitCode') != 255
            or state.get('OOMKilled') is not False or state.get('Running') is not False
            or state.get('Pid') != 0
            or outage.get('result_json_present') is not False
            or outage.get('completion_json_present') is not False
            or outage.get('failed_audit_status') != 'failed'
            or outage.get('failed_audit_trace_complete') is not False
            or outage.get('cause') != 'host_power_outage'
            or outage.get('heartbeat_status') != 'running'
            or outage.get('heartbeat_final_iteration') != FINAL
            or outage.get('last_checkpoint') != heartbeat.get('last_checkpoint')):
        raise ValueError('recovery partial outage facts mismatch')
    resources = outage.get('resources_jsonl', {})
    if (resources.get('sha256') != RESOURCES_SHA256 or resources.get('copied') is not False
            or resources.get('bytes') != RESOURCES_BYTES or resources.get('lines') != RESOURCES_LINES):
        raise ValueError('recovery partial outage resource checksum mismatch')
    detail = str(outage.get('cause_detail', ''))
    skew = outage.get('clock_skew', {})
    if ('172.16.49.208' not in detail or '172.16.50.153' not in detail
            or skew.get('host') != 'pc-411w' or skew.get('reference_host') != 'workpc411'
            or type(skew.get('offset_seconds')) is not int
            or not 8 * 3600 - 600 <= skew['offset_seconds'] <= 8 * 3600 + 600):
        raise ValueError('recovery partial outage cause/clock evidence mismatch')


def _expected(root):
    from .contract import plan as make_plan, production_profile, validate_plan, BATCH
    from .runner import build, native_evidence, COUNTERS
    from . import interrupted
    from sep_research import checkpoints
    import json

    partial = root / 'partial'
    source_plan = read(partial / 'plan.json')
    validate_plan(source_plan)
    if source_plan.get('purpose') != RECOVERY_PURPOSE or digest(source_plan) != PREVIOUS_PLAN_HASH:
        raise ValueError('partial recovery requires the exact interrupted recovery plan')
    if len(source_plan['runs']) != 1:
        raise ValueError('partial recovery requires the single-line recovery plan')
    row = next((r for r in source_plan['runs'] if r['run_id'] == RUN_ID), None)
    if row is None:
        raise ValueError('partial recovery source plan lacks the exact line')
    if (row['family'] != FAMILY or row['training_seed'] != TRAINING_SEED
            or row['start_iteration'] != START or row['target_iteration'] != TARGET
            or row['target_budget'] != 5_000_000 or row['segment_boundaries'] != [0, START, TARGET]
            or row['source_kind'] != 'interrupted_prefix'
            or row['source_checkpoint_id'] != PREFIX_CHECKPOINT_ID
            or row['source_result_hash'] != PREFIX_RESULT_HASH
            or row['source_plan_hash'] != CAMPAIGN_PLAN_HASH
            or row['prefix_audit_complete'] is not False
            or row['original_protocol_eligible'] is not False):
        raise ValueError('partial recovery source row identity/history mismatch')
    record = read(partial / 'preflight.json')
    if (record.get('plan_hash') != PREVIOUS_PLAN_HASH or record.get('run_id') != RUN_ID
            or not record.get('limits_enforced')):
        raise ValueError('partial recovery preflight identity mismatch')
    if read(partial / 'writer.json') != dict(run_id=RUN_ID, attempt_id=PREVIOUS_ATTEMPT,
                                             plan_hash=PREVIOUS_PLAN_HASH):
        raise ValueError('partial recovery writer identity mismatch')
    heartbeat = read(partial / 'heartbeat.json')
    last = heartbeat.get('last_checkpoint', {})
    if (heartbeat.get('run_id') != RUN_ID or heartbeat.get('status') != 'running'
            or heartbeat.get('start_iteration') != START
            or heartbeat.get('segment_boundaries') != [0, START, TARGET]
            or heartbeat.get('final_iteration') != FINAL
            or last.get('iteration') != BOUNDARY or last.get('path') != CHECKPOINT
            or last.get('checkpoint_id') != CHECKPOINT_ID
            or last.get('manifest_hash') != CHECKPOINT_MANIFEST_HASH):
        raise ValueError('partial recovery heartbeat/outage boundary mismatch')
    audit = read(partial / 'failed_audit.json')
    if audit.get('status') != 'failed' or audit.get('trace_complete') is not False:
        raise ValueError('partial recovery failed audit status mismatch')
    _validate_outage(read(partial / 'outage.json'), heartbeat)

    metrics = [json.loads(line) for line in (partial / 'metrics.jsonl').read_text().splitlines()]
    if [m['iteration'] for m in metrics] != list(range(START + 1, FINAL + 1)):
        raise ValueError('partial recovery metric iterations mismatch')
    for metric in metrics:
        if (metric['family'] != row['family'] or metric['training_seed'] != row['training_seed']
                or metric['profile_hash'] != source_plan['profile_hash']):
            raise ValueError('partial recovery metric identity mismatch')
        for key in COUNTERS:
            if metric['counters'][key] != metric['iteration'] * BATCH * (2 if 'agent' in key else 1):
                raise ValueError('partial recovery metric counters mismatch')
    boundary_metric = metrics[BOUNDARY - START - 1]
    if boundary_metric['iteration'] != BOUNDARY:
        raise ValueError('missing partial recovery boundary metric')

    # The failed original prefix at 150 must still validate as the exact chain ancestor.
    nested = interrupted.validate(root / 'source', expected_plan=make_plan(profile=production_profile()))
    if (digest(nested) != PREFIX_RESULT_HASH
            or nested['checkpoint']['checkpoint_id'] != PREFIX_CHECKPOINT_ID):
        raise ValueError('partial recovery chain is not the exact interrupted150 prefix')
    prefix = interrupted.learner_source(root / 'source')

    _, contract = build(row, source_plan['profile'], prefix['contract'])
    bundle = relative(root, CHECKPOINT)
    manifest = checkpoints.validate_bundle(bundle, contract)
    if (manifest['iteration'] != BOUNDARY or manifest['num_env_steps'] != BOUNDARY * BATCH
            or manifest['label'] != CHECKPOINT_LABEL or manifest['joint_checkpoint_id'] != CHECKPOINT_ID
            or sha(bundle / 'manifest.json') != CHECKPOINT_MANIFEST_HASH
            or manifest['initial_role_fingerprints'] != prefix['initial_role_fingerprints']):
        raise ValueError('partial recovery checkpoint identity/counters mismatch')
    if native_evidence(bundle, contract) != boundary_metric['learner_state']:
        raise ValueError('checkpoint 275 native learner state differs from metric')

    files = {f'partial/{name}': sha(partial / name) for name in FILES}
    files.update({CHECKPOINT + '/' + name: value for name, value in tree(bundle).items()})
    files.update({'source/' + name: value for name, value in tree(root / 'source').items()})
    return dict(schema=SCHEMA, run_id=RUN_ID, source_plan_hash=digest(source_plan),
                source_plan_purpose=source_plan['purpose'], source_plan_attempt=PREVIOUS_ATTEMPT,
                source_result_hash=digest(nested), source_container_id=SOURCE_CONTAINER_ID,
                source_image_digest=IMAGE_DIGEST, source_deployment_hash=SOURCE_DEPLOYMENT_HASH,
                campaign_plan_hash=CAMPAIGN_PLAN_HASH,
                checkpoint=dict(path=CHECKPOINT, iteration=BOUNDARY, checkpoint_id=CHECKPOINT_ID,
                                manifest_hash=CHECKPOINT_MANIFEST_HASH, label=CHECKPOINT_LABEL),
                initial_role_fingerprints=deepcopy(manifest['initial_role_fingerprints']),
                segment_boundaries=[0, START, BOUNDARY, TARGET],
                discarded_iterations=list(DISCARDED),
                discarded_completed_env_steps=len(DISCARDED) * BATCH,
                final_computed_iteration=FINAL, prefix_audit_complete=False,
                original_protocol_eligible=False, exact_uninterrupted_resume=False,
                boundary_comparability='heterogeneous_authorized_resume', limitation=LIMITATION,
                files=files)


def validate(path, *, expected_plan=None):
    root = safe(path)
    manifest = read(root / 'partial_recovery.json')
    if manifest.get('schema') != SCHEMA:
        raise ValueError('unsupported partial recovery source schema')
    # Verify the allowlist and bytes BEFORE loading trusted native checkpoint pickle.
    names = set(manifest.get('files', {}))
    mandatory = {f'partial/{name}' for name in FILES}
    if not mandatory <= names or any(name not in mandatory
                                     and not name.startswith(CHECKPOINT + '/')
                                     and not name.startswith('source/') for name in names):
        raise ValueError('invalid partial recovery evidence allowlist')
    if any(sha(relative(root, name)) != value for name, value in manifest['files'].items()):
        raise ValueError('partial recovery evidence hash mismatch')
    if set(tree(root)) != names | {'partial_recovery.json'}:
        raise ValueError('unexpected partial recovery evidence files')
    if expected_plan is not None and read(root / 'partial/plan.json') != expected_plan:
        raise ValueError('partial recovery source plan mismatch')
    if manifest != _expected(root):
        raise ValueError('partial recovery provenance or eligibility mismatch')
    return manifest


def prepare(source, checkpoint, destination, *, expected_plan):
    """Publish a NEW portable evidence directory; never edit the killed attempt."""
    source, checkpoint, destination = map(safe, (source, checkpoint, destination))
    if read(source / 'plan.json') != expected_plan:
        raise ValueError('partial recovery source plan mismatch')
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'partial').mkdir()
    for name in FILES:
        shutil.copyfile(source / name, destination / 'partial' / name)
    target = destination / CHECKPOINT
    target.mkdir(parents=True)
    for name in tree(checkpoint):
        output = relative(target, name)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(relative(checkpoint, name), output)
    # The failed 150 prefix stays in the source history; recovery_partial._expected requires it.
    copy_evidence(source / 'source', destination / 'source')
    write(destination / 'partial_recovery.json', _expected(destination))
    return validate(destination, expected_plan=expected_plan)


def evidence_tree(path):
    root = safe(path)
    manifest = validate(root)
    return dict(manifest['files'], **{'partial_recovery.json': sha(root / 'partial_recovery.json')})


def learner_source(path):
    """Internal restore descriptor, explicitly NOT a cpu_result/completed segment."""
    root = safe(path)
    manifest = validate(root)
    from .runner import native_evidence
    bundle = relative(root, CHECKPOINT)
    contract = read(bundle / 'manifest.json')['contract']
    return dict(source_kind='partial_recovery', contract=contract, endpoint=manifest['checkpoint'],
                initial_role_fingerprints=manifest['initial_role_fingerprints'],
                final_learner_state=native_evidence(bundle, contract),
                discarded_attempt_iterations=len(manifest['discarded_iterations']))
