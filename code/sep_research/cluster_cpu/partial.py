"""Exact, explicitly non-terminal P_G checkpoint-575 prefix of a killed continuation.

attempt_002 continued the original completed306 line under the ready cohort plan and
reached iteration 586, but the host lost power after checkpoint 575 was written.
Iterations 576..586 have no saved bundle and are discarded; this module never presents
the prefix as a completed result, never certifies an uninterrupted or bitwise resume,
and records the outage/clock facts that ended the attempt.
"""
from copy import deepcopy
import shutil

from .common import copy_evidence, digest, read, write, relative, safe, sha, tree

SCHEMA = 'cpu_partial_continuation_v1'
RUN_ID = 'cpu-v3-9539aded8e94-main-P_G-20260914'
FAMILY = 'P_G'
TRAINING_SEED = 20260914
START = 306
BOUNDARY = 575
FINAL = 586
TARGET = 611
DISCARDED = tuple(range(BOUNDARY + 1, FINAL + 1))
CHECKPOINT = 'checkpoints/iteration_000575'
CHECKPOINT_ID = '9862b2bc75e04fd18bcf9e4094a7d929'
CHECKPOINT_MANIFEST_HASH = '347c5c75cef444295420d3171f0ecc419023a643c4570870f278030ed20958eb'
PREVIOUS_PLAN_HASH = '1df32166fe89adc3ab4d5b1bd3478d01fed6653ac5ed781ebe23faf1f972bc78'
COHORT_PURPOSE = 'ready_cohort_10m'
PREVIOUS_ATTEMPT = 'attempt_002'
IMAGE_DIGEST = ('workpc411:5000/traffic-rl-cpu@sha256:'
                '0a23b36462f13c853b6a128f548209c8c37d4a64aa5fe65c8ebd8e6820440f59')
SOURCE_CONTAINER_ID = 'a45646cef41c6534680d5e2d25496f89b8c2b1ff29a30bba6d6c9ae1ec7deb43'
COMPLETED_CHECKPOINT_ID = 'b2bc0942a6824b7592443988e78c6d84'
COMPLETED_RESULT_HASH = 'ea8ccc0659b64c6627404af34b40fd96b949896483e021c0700d5f5a0c23696d'
RESOURCES_SHA256 = 'd45d29358e6840a81db3d2a9ba85ced051ad39d7db2338761f89d92357125201'
RESOURCES_BYTES = 55853968
RESOURCES_LINES = 13348
FILES = ('plan.json', 'preflight.json', 'writer.json', 'metrics.jsonl', 'heartbeat.json',
         'failed_audit.json', 'outage.json')
LIMITATION = ('Killed continuation attempt_002 prefix: learner/critic/Adam/KL/gradient state is verified '
              'only at checkpoint 575, iterations 576..586 were computed but never checkpointed and are '
              'discarded, and environment/cache/episode/sampling RNG restarts deterministically at 575 '
              'under sep_amendment_segment_rng_v1. The line history is heterogeneous (extra authorized '
              'resume boundary); this is not an uninterrupted or bitwise continuation.')


def _validate_outage(outage, heartbeat):
    state = outage.get('host_state', {})
    if (outage.get('schema') != 'cpu_partial_outage_v1' or outage.get('run_id') != RUN_ID
            or outage.get('attempt_id') != PREVIOUS_ATTEMPT
            or outage.get('container_id') != SOURCE_CONTAINER_ID
            or outage.get('image_digest') != IMAGE_DIGEST
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
        raise ValueError('partial outage facts mismatch')
    resources = outage.get('resources_jsonl', {})
    if (resources.get('sha256') != RESOURCES_SHA256 or resources.get('copied') is not False
            or resources.get('bytes') != RESOURCES_BYTES or resources.get('lines') != RESOURCES_LINES):
        raise ValueError('partial outage resource checksum mismatch')
    detail = str(outage.get('cause_detail', ''))
    skew = outage.get('clock_skew', {})
    if ('172.16.49.208' not in detail or '172.16.50.153' not in detail
            or skew.get('host') != 'pc-411w' or skew.get('reference_host') != 'workpc411'
            or type(skew.get('offset_seconds')) is not int
            or not 8 * 3600 - 600 <= skew['offset_seconds'] <= 8 * 3600 + 600):
        raise ValueError('partial outage cause/clock evidence mismatch')


def _expected(root):
    from .contract import plan as make_plan, production_profile, validate_plan, BATCH
    from .runner import build, native_evidence, COUNTERS, validate_result
    from sep_research import checkpoints
    import json

    partial = root / 'partial'
    source_plan = read(partial / 'plan.json')
    validate_plan(source_plan)
    if source_plan.get('purpose') != COHORT_PURPOSE or digest(source_plan) != PREVIOUS_PLAN_HASH:
        raise ValueError('partial source requires the exact ready cohort continuation plan')
    row = next((r for r in source_plan['runs'] if r['run_id'] == RUN_ID), None)
    if row is None:
        raise ValueError('partial source plan lacks the exact line')
    if (row['family'] != FAMILY or row['training_seed'] != TRAINING_SEED
            or row['start_iteration'] != START or row['target_iteration'] != TARGET
            or row['target_budget'] != 10_000_000 or row['segment_boundaries'] != [0, START, TARGET]
            or row['source_checkpoint_id'] != COMPLETED_CHECKPOINT_ID
            or row['source_result_hash'] != COMPLETED_RESULT_HASH):
        raise ValueError('partial source row identity/history mismatch')
    record = read(partial / 'preflight.json')
    if (record.get('plan_hash') != PREVIOUS_PLAN_HASH or record.get('run_id') != RUN_ID
            or not record.get('limits_enforced')):
        raise ValueError('partial preflight identity mismatch')
    if read(partial / 'writer.json') != dict(run_id=RUN_ID, attempt_id=PREVIOUS_ATTEMPT,
                                             plan_hash=PREVIOUS_PLAN_HASH):
        raise ValueError('partial writer identity mismatch')
    heartbeat = read(partial / 'heartbeat.json')
    last = heartbeat.get('last_checkpoint', {})
    if (heartbeat.get('run_id') != RUN_ID or heartbeat.get('status') != 'running'
            or heartbeat.get('start_iteration') != START
            or heartbeat.get('segment_boundaries') != [0, START, TARGET]
            or heartbeat.get('final_iteration') != FINAL
            or last.get('iteration') != BOUNDARY or last.get('path') != CHECKPOINT
            or last.get('checkpoint_id') != CHECKPOINT_ID
            or last.get('manifest_hash') != CHECKPOINT_MANIFEST_HASH):
        raise ValueError('partial heartbeat/outage boundary mismatch')
    audit = read(partial / 'failed_audit.json')
    if audit.get('status') != 'failed' or audit.get('trace_complete') is not False:
        raise ValueError('partial failed audit status mismatch')
    _validate_outage(read(partial / 'outage.json'), heartbeat)

    metrics = [json.loads(line) for line in (partial / 'metrics.jsonl').read_text().splitlines()]
    if [m['iteration'] for m in metrics] != list(range(START + 1, FINAL + 1)):
        raise ValueError('partial metric iterations mismatch')
    for metric in metrics:
        if (metric['family'] != row['family'] or metric['training_seed'] != row['training_seed']
                or metric['profile_hash'] != source_plan['profile_hash']):
            raise ValueError('partial metric identity mismatch')
        for key in COUNTERS:
            if metric['counters'][key] != metric['iteration'] * BATCH * (2 if 'agent' in key else 1):
                raise ValueError('partial metric counters mismatch')
    boundary_metric = metrics[BOUNDARY - START - 1]
    if boundary_metric['iteration'] != BOUNDARY:
        raise ValueError('missing partial boundary metric')

    # The original completed306 prefix must still validate as the exact chain ancestor.
    nested = validate_result(root / 'source', expected_plan=make_plan(profile=production_profile()),
                             run_id=RUN_ID)
    if (digest(nested) != COMPLETED_RESULT_HASH or nested['final_iteration'] != START
            or nested['endpoint']['checkpoint_id'] != COMPLETED_CHECKPOINT_ID):
        raise ValueError('partial source chain is not the exact completed306 original')

    _, contract = build(row, source_plan['profile'], nested['contract'])
    bundle = relative(root, CHECKPOINT)
    manifest = checkpoints.validate_bundle(bundle, contract)
    if (manifest['iteration'] != BOUNDARY or manifest['num_env_steps'] != BOUNDARY * BATCH
            or manifest['label'] != 'incomplete' or manifest['joint_checkpoint_id'] != CHECKPOINT_ID
            or sha(bundle / 'manifest.json') != CHECKPOINT_MANIFEST_HASH
            or manifest['initial_role_fingerprints'] != nested['initial_role_fingerprints']):
        raise ValueError('partial checkpoint identity/counters mismatch')
    if native_evidence(bundle, contract) != boundary_metric['learner_state']:
        raise ValueError('checkpoint 575 native learner state differs from metric')

    files = {f'partial/{name}': sha(partial / name) for name in FILES}
    files.update({CHECKPOINT + '/' + name: value for name, value in tree(bundle).items()})
    files.update({'source/' + name: value for name, value in tree(root / 'source').items()})
    return dict(schema=SCHEMA, run_id=RUN_ID, source_plan_hash=digest(source_plan),
                source_plan_purpose=source_plan['purpose'], source_plan_attempt=PREVIOUS_ATTEMPT,
                source_result_hash=digest(nested),
                source_container_id=SOURCE_CONTAINER_ID, source_image_digest=IMAGE_DIGEST,
                checkpoint=dict(path=CHECKPOINT, iteration=BOUNDARY, checkpoint_id=CHECKPOINT_ID,
                                manifest_hash=CHECKPOINT_MANIFEST_HASH, label='incomplete'),
                initial_role_fingerprints=deepcopy(manifest['initial_role_fingerprints']),
                segment_boundaries=[0, START, BOUNDARY, TARGET],
                discarded_iterations=list(DISCARDED),
                discarded_completed_env_steps=len(DISCARDED) * BATCH,
                final_computed_iteration=FINAL, prefix_audit_complete=True,
                original_protocol_eligible=True, exact_uninterrupted_resume=False,
                boundary_comparability='heterogeneous_authorized_resume', limitation=LIMITATION,
                files=files)


def validate(path, *, expected_plan=None):
    root = safe(path)
    manifest = read(root / 'partial.json')
    if manifest.get('schema') != SCHEMA:
        raise ValueError('unsupported partial continuation source schema')
    # Verify the allowlist and bytes BEFORE loading trusted native checkpoint pickle.
    names = set(manifest.get('files', {}))
    mandatory = {f'partial/{name}' for name in FILES}
    if not mandatory <= names or any(name not in mandatory
                                     and not name.startswith(CHECKPOINT + '/')
                                     and not name.startswith('source/') for name in names):
        raise ValueError('invalid partial continuation evidence allowlist')
    if any(sha(relative(root, name)) != value for name, value in manifest['files'].items()):
        raise ValueError('partial continuation evidence hash mismatch')
    if set(tree(root)) != names | {'partial.json'}:
        raise ValueError('unexpected partial continuation evidence files')
    if expected_plan is not None and read(root / 'partial/plan.json') != expected_plan:
        raise ValueError('partial continuation source plan mismatch')
    if manifest != _expected(root):
        raise ValueError('partial continuation provenance or eligibility mismatch')
    return manifest


def prepare(source, checkpoint, destination, *, expected_plan):
    """Publish a NEW portable evidence directory; never edit the killed attempt."""
    source, checkpoint, destination = map(safe, (source, checkpoint, destination))
    if read(source / 'plan.json') != expected_plan:
        raise ValueError('partial continuation source plan mismatch')
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
    # The completed306 chain stays in the source history; partial._expected requires it.
    copy_evidence(source / 'source', destination / 'source')
    write(destination / 'partial.json', _expected(destination))
    return validate(destination, expected_plan=expected_plan)


def evidence_tree(path):
    root = safe(path)
    manifest = validate(root)
    return dict(manifest['files'], **{'partial.json': sha(root / 'partial.json')})


def learner_source(path):
    """Internal restore descriptor, explicitly NOT a cpu_result/completed segment."""
    root = safe(path)
    manifest = validate(root)
    from .runner import native_evidence
    bundle = relative(root, CHECKPOINT)
    contract = read(bundle / 'manifest.json')['contract']
    return dict(source_kind='partial_continuation', contract=contract, endpoint=manifest['checkpoint'],
                initial_role_fingerprints=manifest['initial_role_fingerprints'],
                final_learner_state=native_evidence(bundle, contract),
                discarded_attempt_iterations=len(manifest['discarded_iterations']))
