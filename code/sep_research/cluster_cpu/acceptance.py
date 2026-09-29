"""Evidence-based CPU acceptance. Operator observations are integrity-bound inputs.

This verifier does not authenticate operator Docker/host observations. It derives
checks from concrete inspected configurations and measured result files; a passed
boolean is never sufficient. No acceptance receipt is supplied by this release.
"""
import argparse
import json
from pathlib import Path

from .common import read, write, digest, sha, relative
from .contract import FAMILIES, PROFILE, validate_profile


def verify(path, profile):
    from .runner import validate_result
    path = Path(path)
    value = read(path)
    if value.get('schema') != 'cpu_acceptance_v1' or value['profile_hash'] != validate_profile(profile):
        raise ValueError('acceptance profile mismatch')
    results = {}
    observations = []
    for entry in value['evidence']:
        result_path = relative(path.parent, entry['result'])
        result = validate_result(result_path)
        if digest(result) != entry['result_hash'] or result['profile_hash'] != value['profile_hash']:
            raise ValueError('acceptance result hash/profile mismatch')
        if not result['limits_enforced'] or result['image_digest'].endswith('0'*64):
            raise ValueError('diagnostic smoke is not acceptance evidence')
        observation_path = relative(path.parent, entry['observation'])
        if sha(observation_path) != entry['observation_hash']:
            raise ValueError('operator observation hash mismatch')
        observation = read(observation_path)
        inspect = observation['docker_inspect']
        cfg, host, state = inspect['Config'], inspect['HostConfig'], inspect['State']
        if (cfg['Image'] != result['image_digest'] or state['Running'] or state['OOMKilled']
                or state['ExitCode'] != 0 or host['NetworkMode'] != 'none'
                or host['Memory'] != profile['memory_limit_bytes']
                or host['MemorySwap'] != profile['memory_limit_bytes']
                or host['NanoCpus'] != profile['cpu_quota_logical'] * 10**9
                or host['ShmSize'] < profile['shm_bytes']
                or host.get('DeviceRequests') or host.get('Privileged')
                or host.get('CpusetCpus') or host.get('CpusetMems')):
            raise ValueError('actual offline Docker CPU limits/exit not accepted')
        samples = [json.loads(line) for line in (result_path/'resources.jsonl').read_text().splitlines()]
        samples.append(result['final_resources'])
        if (max(s['cgroup']['peak_bytes'] for s in samples) > .9*profile['memory_limit_bytes']
                or any(s['cgroup']['swap_current_bytes'] for s in samples)
                or min(s['host_mem_available_bytes'] for s in samples) < observation['host_reserve_bytes']):
            raise ValueError('insufficient observed memory reserve or swap use')
        # Policy rewards are deliberately never read for acceptance selection.
        if observation['spill_review']['result_hash'] != entry['result_hash']:
            raise ValueError('spill review is not bound to this run')
        if observation['spill_review']['classification'] != 'no_sustained_thrashing':
            raise ValueError('spill/restore throughput review unresolved')
        key = entry['result_hash']
        results[key] = result
        observations.append((result, observation, result_path))
    pilots = [(r,o,p) for r,o,p in observations if read(p/'plan.json')['mode'] == 'pilot']
    for family in FAMILIES:
        matches = [(r,o,p) for r,o,p in pilots if r['row']['family'] == family
                   and r['final_iteration']-r['start_iteration'] >= 10]
        if not matches:
            raise ValueError('missing ten-iteration family pilot: '+family)
        for r,o,p in matches:
            # Production conditions are exposed by original episode callbacks.
            metrics = [json.loads(line) for line in (p/'metrics.jsonl').read_text().splitlines()]
            keys = set()
            for metric in metrics:
                raw = metric['result']
                custom = raw.get('env_runners', {}).get('custom_metrics', raw.get('custom_metrics', {}))
                keys.update(k.removesuffix('_mean') for k in custom if k.startswith('v_pv/') and k.endswith('_mean'))
            from snfs_traffic.rl.centralized_pv_env import _condition_key
            expected = {'v_pv/'+_condition_key(d,f) for d in (.1,.2,.3,.4) for f in (.9,.95,1.)}
            census = read(p/'seed_audit.json')
            if not expected <= keys or not all(any(rec['warmup_cache_hit'] for rec in env['records'])
                                              for env in census['environments'] if env['measurement_steps']):
                raise ValueError('pilot has not filled warm caches and covered training conditions')
            if not any(child['row']['source_result_hash'] == digest(r)
                       and child['restoration']['accepted_iteration'] == r['final_iteration']
                       for child in results.values() if child['source']):
                raise ValueError('pilot lacks measured stop/copy/restore/update evidence')
    if not any(o['host_logical_cpus'] == 36 for _,o,_ in pilots):
        raise ValueError('missing single-job 36-logical-CPU pilot')
    pairs = [(a,b) for a in pilots for b in pilots if a is not b
             and a[0]['row']['family'].endswith('_G') and b[0]['row']['family'].endswith('_G')
             and a[1]['host_id'] == b[1]['host_id'] and a[1]['host_logical_cpus'] == b[1]['host_logical_cpus'] == 72
             and a[1]['container_id'] != b[1]['container_id']
             and min(a[1]['training_finished_unix'],b[1]['training_finished_unix'])
                 > max(a[1]['training_started_unix'],b[1]['training_started_unix'])]
    if not pairs:
        raise ValueError('missing concurrent graph jobs on 72 logical CPUs')
    if not any(read(p/'plan.json')['mode'] == 'smoke' and o['initial_output_entries'] == []
               for _,o,p in observations):
        raise ValueError('missing offline empty-output image smoke')
    return value


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', required=True)
    p.add_argument('--profile', help='JSON CPU profile, default candidate')
    args = p.parse_args(argv)
    value = verify(args.evidence, read(args.profile) if args.profile else PROFILE)
    print(json.dumps(dict(status='accepted', profile_hash=value['profile_hash'], acceptance_hash=digest(value))))


if __name__ == '__main__':
    main()
