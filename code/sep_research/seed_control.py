"""Frozen training seed exclusion; original reset owns draw order and physics.

The sole RNG exception is rejection sampling of a reserved raw draw, using the
original _next_seed again at that position, before condition sampling. Audit
records use protocol.SEED_FIELDS order (which differs from reset draw order).
Snapshot installation deliberately never enters this training boundary.
"""
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import numpy as np

from sep_research import protocol
from snfs_traffic.rl.centralized_pv_env import CentralizedPvEnv

MANIFEST_DIR = Path(__file__).resolve().parent / 'manifests'
MANIFEST_NAMES = ('validation_seeds.json', 'test_seeds.json', 'development_seeds.json')


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def frozen_seed_guard():
    """Read authoritative files and require every published canonical row."""
    manifests, hashes, counts, values = [], {}, {}, set()
    for split, root, name in zip(
            ('validation', 'test', 'development'),
            (protocol.VALIDATION_ROOT, protocol.TEST_ROOT, protocol.DEVELOPMENT_ROOT),
            MANIFEST_NAMES):
        raw = (MANIFEST_DIR / name).read_bytes()
        try:
            rows = tuple(protocol.SeedRow(**row) for row in json.loads(raw))
        except (TypeError, KeyError) as exc:
            raise ValueError(f'invalid reserved manifest: {name}') from exc
        protocol.validate_seed_manifest(rows)
        kwargs = dict(density_tokens=(30,), fraction_tokens=(100,), runs=20) if split == 'validation' else {}
        expected = protocol.seed_manifest(split, root, **kwargs)
        if set(rows) != set(expected):
            raise ValueError(f'incomplete or altered published manifest: {name}')
        manifests.append(rows)
        hashes[name] = hashlib.sha256(raw).hexdigest()
        counts[split] = len(rows)
        values.update(getattr(row, field) for row in rows for field in protocol.SEED_FIELDS)
    protocol.validate_split_separation(*manifests)
    guard = dict(schema='sep_seed_guard_v1', seed_fields=list(protocol.SEED_FIELDS),
                 reserved_values=sorted(values), manifest_sha256=hashes, manifest_rows=counts,
                 collision_rule='reject_raw_draw_with_original_next_seed_before_condition')
    guard['fingerprint'] = hashlib.sha256(_json(guard).encode()).hexdigest()
    return guard


def validate_seed_guard(guard):
    expected = frozen_seed_guard()
    try:
        valid = _json(guard) == _json(expected)
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError('seed guard does not match frozen manifests/fingerprint/types')
    return frozenset(expected['reserved_values'])


def _seed(value, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < 0):
        raise ValueError(f'{name} must be a nonnegative integer')
    return int(value)


class GuardedCentralizedPvEnv(CentralizedPvEnv):
    def __init__(self, config):
        self._reserved_seeds = validate_seed_guard(config['seed_guard'])
        _seed(config.get('seed', 0), 'seed')
        self._seed_audit = dict(
            guard_hash=config['seed_guard']['fingerprint'],
            worker_index=int(getattr(config, 'worker_index', 0)),
            vector_index=int(getattr(config, 'vector_index', 0)),
            recreated_worker=bool(getattr(config, 'recreated_worker', False)),
            instance_id=uuid4().hex, records=[], rejected_values=[], failed_resets=0,
            measurement_steps=0, warmup_steps=0, episode_measurement_steps=0,
            pending_transitions=0)
        self._training_measurement_active = False
        self._counting_warmup = False
        # Keep the actual EnvContext, including worker/vector seed offsets.
        super().__init__(config)

    def _new_simulator(self, *, scenario_seed):
        simulator = super()._new_simulator(scenario_seed=scenario_seed)
        original_step = simulator.step
        def warmup_step(*args, **kwargs):
            state = original_step(*args, **kwargs)
            if self._counting_warmup:
                self._seed_audit['warmup_steps'] += 1
            return state
        simulator.step = warmup_step
        return simulator

    def _next_seed(self):
        value = super()._next_seed()
        while value in self._reserved_seeds:
            self._seed_audit['rejected_values'].append(value)
            value = super()._next_seed()
        return value

    def reset(self, *, seed=None, options=None):
        # Validate all overrides before even Gym's RNG reseed/cache clearing.
        if seed is not None:
            _seed(seed, 'seed')
        if options is not None and not isinstance(options, Mapping):
            raise ValueError('reset options must be a mapping')
        for name in protocol.SEED_FIELDS:
            if options is not None and name in options:
                value = _seed(options[name], name)
                if value in self._reserved_seeds:
                    raise ValueError(f'reserved explicit {name}={value}')
        start = len(self._seed_audit['rejected_values'])
        warm_before = self._seed_audit['warmup_steps']
        self._counting_warmup = True
        try:
            observation, info = super().reset(seed=seed, options=options)
            actual = [_seed(info[field], field) for field in protocol.SEED_FIELDS]
            if any(value in self._reserved_seeds for value in actual):
                raise ValueError('reserved actual reset seed (including warmup cache)')
            executed = self._seed_audit['warmup_steps'] - warm_before
            if (type(info['warmup_steps']) is not int or info['warmup_steps'] != self._warmup_steps
                    or type(info['warmup_cache_hit']) is not bool
                    or executed != (0 if info['warmup_cache_hit'] else info['warmup_steps'])):
                raise ValueError('warmup execution disagrees with reset info')
        except BaseException:
            self._seed_audit['failed_resets'] += 1
            raise
        finally:
            self._counting_warmup = False
        self._seed_audit['records'].append(dict(
            seeds=actual, rejected_values=self._seed_audit['rejected_values'][start:],
            warmup_cache_hit=info['warmup_cache_hit'], warmup_steps=executed,
            requested_warmup_steps=info['warmup_steps'],
            measurement_steps_before=self._seed_audit['measurement_steps'],
            previous_episode_steps=self._seed_audit['episode_measurement_steps']))
        self._seed_audit['episode_measurement_steps'] = 0
        self._training_measurement_active = True
        return observation, info

    def get_seed_audit(self):
        audit = deepcopy(self._seed_audit)
        audit['rejected_count'] = len(audit['rejected_values'])
        return audit


def validate_seed_audit(payload, guard, *, expected_workers, envs_per_worker=1,
                        create_local_env_runner=True, require_complete=True):
    """Validate v2 evidence without Ray or simulation; v1 cannot prove lifecycle.

    Expected worker geometry comes from the caller's contract, not the payload.
    This certifies seed/counter continuity only, never a main budget or profile.
    """
    reserved = validate_seed_guard(guard)
    def integer(value):
        if type(value) is not int or value < 0:
            raise ValueError('seed audit invalid integer type/count')
        return value

    integer(expected_workers)
    if integer(envs_per_worker) < 1 or type(create_local_env_runner) is not bool:
        raise ValueError('seed audit invalid expected geometry')
    if type(require_complete) is not bool:
        raise ValueError('seed audit invalid completeness request')
    worker_ids = set(range(expected_workers + 1))
    env_workers = worker_ids if create_local_env_runner or not expected_workers else worker_ids - {0}
    env_ids = {(w, v) for w in env_workers for v in range(envs_per_worker)}

    def census(value):
        if (value['schema'] != 'sep_seed_audit_v2'
                or value['seed_fields'] != guard['seed_fields']
                or value['manifest_sha256'] != guard['manifest_sha256']
                or value['guard_hash'] != guard['fingerprint']):
            raise ValueError('seed audit invalid schema/guard hash')
        workers, audits = value['workers'], value['environments']
        if type(workers) is not list or type(audits) is not list:
            raise ValueError('seed audit invalid census types')
        ids, actors = [], []
        for worker in workers:
            ids.append(integer(worker['worker_index']))
            if worker['recreated'] is not False or worker['reconstructed'] is not False:
                raise ValueError('seed audit restart lifecycle')
            actor = worker['actor_id']
            if worker['worker_index'] == 0:
                if actor is not None:
                    raise ValueError('seed audit local actor identity')
            elif type(actor) is not str or not actor:
                raise ValueError('seed audit actor identity')
            else:
                if len(actor) != 32 or any(c not in '0123456789abcdef' for c in actor):
                    raise ValueError('seed audit actor incarnation format')
                actors.append(actor)
            if 'runtime' in worker:
                runtime = worker['runtime']
                if (integer(runtime['pid']) == 0 or integer(runtime['torch_threads']) == 0
                        or type(runtime['node_id']) is not str or not runtime['node_id']
                        or (runtime['cuda_visible_devices'] is not None
                            and type(runtime['cuda_visible_devices']) is not str)):
                    raise ValueError('seed audit invalid worker runtime')
                resources = runtime['assigned_resources']
                if (type(resources) is not dict
                        or any(type(k) is not str or type(v) not in (int, float)
                               or not np.isfinite(v) or v < 0 for k, v in resources.items())):
                    raise ValueError('seed audit invalid assigned resources')
                threads = runtime['thread_environment']
                if (type(threads) is not dict
                        or set(threads) != {'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'}
                        or any(v is not None and type(v) is not str for v in threads.values())):
                    raise ValueError('seed audit invalid thread environment')
        healthy = value['healthy_worker_ids']
        if (type(healthy) is not list or any(type(w) is not int for w in healthy)
                or len(healthy) != expected_workers or set(healthy) != worker_ids - {0}
                or len(ids) != len(worker_ids) or set(ids) != worker_ids
                or len(set(actors)) != len(actors)
                or integer(value['remote_worker_restarts']) != 0):
            raise ValueError('seed audit incomplete worker retrieval/restart lifecycle')
        seen, instances = set(), set()
        for audit in audits:
            key = (integer(audit['worker_index']), integer(audit['vector_index']))
            identity = audit['instance_id']
            if (key in seen or type(identity) is not str or not identity or identity in instances):
                raise ValueError('seed audit duplicate environment incarnation')
            if len(identity) != 32 or any(c not in '0123456789abcdef' for c in identity):
                raise ValueError('seed audit environment incarnation format')
            seen.add(key)
            instances.add(identity)
            if (audit['guard_hash'] != guard['fingerprint'] or audit['recreated_worker'] is not False
                    or integer(audit['failed_resets']) or integer(audit['pending_transitions'])):
                raise ValueError('seed audit invalid guard/reset/transition lifecycle')
            records = audit['records']
            if type(records) is not list or not records:
                raise ValueError('seed audit missing reset records')
            rejected, warm, previous = [], 0, 0
            if records[0]['measurement_steps_before'] != 0 or records[0]['previous_episode_steps'] != 0:
                raise ValueError('seed audit nonzero construction counters')
            for record in records:
                seeds = record['seeds']
                if (type(seeds) is not list or len(seeds) != 5
                        or any(integer(v) in reserved for v in seeds)):
                    raise ValueError('seed audit invalid/reserved actual seed')
                if type(record['rejected_values']) is not list:
                    raise ValueError('seed audit invalid rejection types')
                rejected.extend(record['rejected_values'])
                if type(record['warmup_cache_hit']) is not bool:
                    raise ValueError('seed audit invalid cache hit type')
                executed = integer(record['warmup_steps'])
                requested = integer(record['requested_warmup_steps'])
                if (requested != records[0]['requested_warmup_steps']
                        or executed != (0 if record['warmup_cache_hit'] else requested)):
                    raise ValueError('seed audit inconsistent warmup work')
                warm += executed
                before = integer(record['measurement_steps_before'])
                if before != previous + integer(record['previous_episode_steps']):
                    raise ValueError('seed audit inconsistent reset step boundaries')
                previous = before
            if (integer(audit['measurement_steps']) != previous + integer(audit['episode_measurement_steps'])
                    or integer(audit['warmup_steps']) != warm):
                raise ValueError('seed audit inconsistent physical counters')
            if (type(audit['rejected_values']) is not list or rejected != audit['rejected_values']
                    or len(rejected) != integer(audit['rejected_count'])
                    or any(integer(v) not in reserved for v in rejected)):
                raise ValueError('seed audit invalid rejection accounting')
        if len(audits) != len(env_ids) or seen != env_ids:
            raise ValueError('seed audit incomplete worker/vector census')
        for name, total in (
                ('reset_count', sum(len(a['records']) for a in audits)),
                ('rejected_count', sum(a['rejected_count'] for a in audits)),
                ('measurement_steps', sum(a['measurement_steps'] for a in audits)),
                ('warmup_steps', sum(a['warmup_steps'] for a in audits))):
            if integer(value[name]) != total:
                raise ValueError('seed audit inconsistent aggregate counts')

    try:
        census(payload)
        initial = payload.get('initial')
        if initial is not None:
            census(initial)
            if initial.get('initial') is not None:
                raise ValueError('seed audit nested initial census')
            old = {(a['worker_index'], a['vector_index']): a for a in initial['environments']}
            if _json(payload['workers']) != _json(initial['workers']):
                raise ValueError('seed audit actor incarnation changed')
            for audit in payload['environments']:
                prior = old[(audit['worker_index'], audit['vector_index'])]
                if audit['instance_id'] != prior['instance_id']:
                    raise ValueError('seed audit environment incarnation changed')
                if _json(audit['records'][:len(prior['records'])]) != _json(prior['records']):
                    raise ValueError('seed audit lost prior reset records')
                if any(audit[k] < prior[k] for k in ('measurement_steps', 'warmup_steps')):
                    raise ValueError('seed audit counters regressed')
                if (len(audit['records']) > len(prior['records'])
                        and audit['records'][len(prior['records'])]['measurement_steps_before']
                        < prior['measurement_steps']):
                    raise ValueError('seed audit reset predates initial counters')
        if 'resumed' in payload and type(payload['resumed']) is not bool:
            raise ValueError('seed audit invalid resumed type')
        segments = payload.get('prior_audit_segments', [])
        if type(segments) is not list:
            raise ValueError('seed audit invalid prior segments')
        names = set()
        for segment in segments:
            name, digest = segment['file'], segment['sha256']
            if (type(name) is not str or name in ('.', '..') or Path(name).name != name or name in names
                    or type(digest) is not str or len(digest) != 64
                    or any(c not in '0123456789abcdef' for c in digest)):
                raise ValueError('seed audit invalid prior file/hash')
            names.add(name)
        if require_complete and (payload.get('resumed') is not False or segments
                                 or initial is None or payload.get('trace_complete') is not True
                                 or payload.get('status') != 'certified_seed_exclusion'):
            raise ValueError('seed audit missing fresh complete lifecycle')
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('seed audit missing/invalid evidence types') from exc
    return dict(measurement_steps=payload['measurement_steps'], warmup_steps=payload['warmup_steps'],
                reset_count=payload['reset_count'], rejected_count=payload['rejected_count'])
