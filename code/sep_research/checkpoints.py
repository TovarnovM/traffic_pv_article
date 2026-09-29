"""Trusted-local atomic whole-Algorithm bundles, schema sep_pair_v2.

Public API: save_bundle, validate_bundle, restore_bundle, load_inference_weights. The bundle contains
manifest.json and algorithm/ (RLlib native save, including both policy states
and optimizer slots). No independent role selection. Hash validation precedes
pickle restore; hashes provide integrity, not authentication of external data.
Numeric exports never require pickle or Ray. Supply the original frozen training
contract, including source provenance, even for CPU deployment of GPU weights.
Trusted-local integrity does not authenticate forged manifests/external models.
Development v1 bundles are not converted.
Resume is explicitly non-bitwise: worker/environment/RNG/cache state is absent.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid
from zipfile import BadZipFile

import numpy as np

SCHEMA = 'sep_pair_v2'
POLICY_IDS = ('pv_policy', 'av_policy')


@contextmanager
def _directory_fd(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        yield fd
    finally:
        os.close(fd)


def fingerprint(value):
    """Stable recursive digest for weights and native optimizer state."""
    digest = hashlib.sha256()
    def visit(item):
        if hasattr(item, 'detach'):
            item = item.detach().cpu().numpy()
        if isinstance(item, np.ndarray):
            digest.update(str(item.dtype).encode())
            digest.update(str(item.shape).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                visit(str(key))
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(b'[')
            for element in item:
                visit(element)
            digest.update(b']')
        else:
            digest.update(json.dumps(item, sort_keys=True, default=str).encode() + b'\0')
    visit(value)
    return digest.hexdigest()


def compatible(saved, expected, allow_runtime_changes=False):
    saved, expected = dict(saved), dict(expected)
    if allow_runtime_changes:
        saved.pop('resources', None)
        expected.pop('resources', None)
        for config in (saved, expected):
            if 'rllib' in config:
                config['rllib'] = dict(config['rllib'])
                for key in ('num_env_runners', 'num_cpus_per_env_runner', 'num_gpus'):
                    config['rllib'].pop(key, None)
    if saved != expected:
        raise ValueError('checkpoint configuration/source/schema mismatch')


def _hashes(root):
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('symlinks are forbidden in checkpoint bundles')
        if path.is_file() and path != root / 'manifest.json':
            result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _states(algorithm, contract):
    from ray.rllib.policy.policy import PolicySpec
    config = algorithm.config
    if (config.env_config != contract['env']
            or list(config.policies_to_train) != contract['trainable_ids']
            or any(getattr(config, key) != value for key, value in contract['ppo'].items())):
        raise ValueError('Algorithm does not match the declared environment/PPO contract')
    for pid, spec in config.policies.items():
        spec = PolicySpec(*spec) if isinstance(spec, tuple) else spec
        if (pid not in contract['models'] or spec.config != contract['models'][pid]
                or str(spec.observation_space) != contract['spaces'][pid]['observation']
                or str(spec.action_space) != contract['spaces'][pid]['action']):
            raise ValueError('Algorithm does not match the declared model/space contract')
    if set(algorithm.env_runner.policy_map) != set(POLICY_IDS):
        raise ValueError('Algorithm must own exactly the policy pair')
    result = {}
    for pid in POLICY_IDS:
        policy = algorithm.get_policy(pid)
        state = policy.get_state()
        optimizers = state.get('_optimizer_variables', [])
        trainable = pid in contract['trainable_ids']
        if trainable and not optimizers:
            raise ValueError(f'missing optimizer ownership for {pid}')
        if not trainable and (optimizers or policy.get_weights()):
            raise ValueError('native role has learned state')
        result[pid] = dict(weights=fingerprint(policy.get_weights()),
                           optimizers=fingerprint(optimizers),
                           optimizer_count=len(optimizers),
                           # Connector caches are outside non-bitwise resume.
                           state=fingerprint({key: value for key, value in state.items()
                               if key not in ('connector_configs', 'policy_spec')}))
    return result



def _numeric_weights(weights):
    if not isinstance(weights, dict) or not weights:
        raise ValueError('missing numeric policy weights')
    result = {}
    for key, value in weights.items():
        if not isinstance(key, str) or not key or '/' in key:
            raise ValueError('invalid numeric weight key')
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        if value.dtype.kind != 'f' or not np.isfinite(value).all():
            raise ValueError(f'invalid numeric weight {key}: expected finite floating dtype')
        result[key] = value.copy()
    return result


def _export_metadata(manifest, pid, weights):
    return dict(joint_checkpoint_id=manifest['joint_checkpoint_id'], policy_id=pid,
        role='pv' if pid == 'pv_policy' else 'av', family=manifest['family'],
        encoder=manifest['pv_encoder'] if pid == 'pv_policy' else 'graph_mlp',
        weights_fingerprint=manifest['role_states'][pid]['weights'],
        file=f'weights/{pid}.npz',
        arrays={key: dict(shape=list(value.shape), dtype=str(value.dtype))
                for key, value in weights.items()})


def _read_inference_weights(path, manifest):
    exports = manifest.get('inference_weights')
    if not isinstance(exports, dict) or set(exports) != set(POLICY_IDS):
        raise ValueError('missing numeric policy pair')
    expected_files = {f'weights/{pid}.npz' for pid in manifest['trainable_ids']}
    actual_files = {str(p.relative_to(path)) for p in (path / 'weights').rglob('*') if p.is_file()}
    if actual_files != expected_files:
        raise ValueError('missing or orphan numeric exports')
    result = {}
    for pid in POLICY_IDS:
        entry = exports[pid]
        if pid not in manifest['trainable_ids']:
            if entry is not None or manifest['role_states'][pid]['weights'] != fingerprint({}):
                raise ValueError('native role has numeric weights')
            result[pid] = None
            continue
        if not isinstance(entry, dict) or entry.get('file') != f'weights/{pid}.npz':
            raise ValueError('missing or mixed numeric export metadata')
        try:
            with np.load(path / entry['file'], allow_pickle=False) as archive:
                if len(archive.files) != len(set(archive.files)):
                    raise ValueError('duplicate numeric weight keys')
                weights = _numeric_weights(dict(archive))
        except (OSError, ValueError, TypeError, EOFError, BadZipFile) as exc:
            raise ValueError(f'invalid numeric export for {pid}') from exc
        if entry != _export_metadata(manifest, pid, weights):
            raise ValueError('numeric export metadata mismatch')
        if fingerprint(weights) != manifest['role_states'][pid]['weights']:
            raise ValueError('numeric role weight fingerprint mismatch')
        from sep_research.inference import build_av_model, build_pv_model
        try:
            if pid == 'pv_policy':
                build_pv_model(weights, manifest['pv_encoder'])
            else:
                build_av_model(weights, manifest['contract']['env'])
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f'numeric model/schema mismatch for {pid}') from exc
        result[pid] = weights
    return result


def load_inference_weights(path, expected):
    """Return ({policy_id: CPU NumPy weights or native None}, manifest).

    Strictly validate the whole trusted-local bundle first. Never unpickle or
    import Ray. Supply the frozen training contract for GPU-to-CPU deployment;
    no runtime/provenance bypass is offered.
    """
    manifest = validate_bundle(path, expected)
    return _read_inference_weights(Path(path), manifest), manifest

def save_bundle(algorithm, path, contract, *, num_env_steps, iteration,
                label, initial_fingerprints):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if type(num_env_steps) is not int or num_env_steps < 0 or type(iteration) is not int or iteration < 0:
        raise ValueError('invalid counters')
    if label not in ('candidate', 'final', 'incomplete', 'tiny'):
        raise ValueError('invalid checkpoint label')
    if label == 'final' and (num_env_steps < contract['budget'] or contract['profile'] != 'main'):
        raise ValueError('final requires completed main budget')
    if (algorithm._counters['num_env_steps_sampled'] != num_env_steps
            or algorithm.iteration != iteration):
        raise ValueError('save counters must match the actual Algorithm')
    if set(initial_fingerprints) != set(POLICY_IDS):
        raise ValueError('missing initial pair fingerprints')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f'.{path.name}.partial_', dir=path.parent))
    states = _states(algorithm, contract)
    (temp / 'algorithm').mkdir()
    algorithm.save_checkpoint(str(temp / 'algorithm'))
    manifest = dict(schema=SCHEMA, joint_checkpoint_id=uuid.uuid4().hex,
        contract=contract, policy_ids=list(POLICY_IDS), trainable_ids=contract['trainable_ids'],
        family=contract['family'], pv_encoder=contract['pv_encoder'],
        num_env_steps=num_env_steps, iteration=iteration, label=label,
        budget_reached=num_env_steps >= contract['budget'],
        resume_semantics='algorithm_state_non_bitwise', eligible_for_main_completeness=False,
        initial_role_fingerprints=initial_fingerprints, role_states=states,
        model_space_fingerprint=fingerprint({'models': contract['models'], 'spaces': contract['spaces']}),
        inference_weights={})
    (temp / 'weights').mkdir()
    for pid in POLICY_IDS:
        if pid not in contract['trainable_ids']:
            manifest['inference_weights'][pid] = None
            continue
        weights = _numeric_weights(algorithm.get_policy(pid).get_weights())
        if fingerprint(weights) != states[pid]['weights']:
            raise ValueError('policy weights changed during checkpoint save')
        entry = _export_metadata(manifest, pid, weights)
        np.savez(temp / entry['file'], **weights)
        manifest['inference_weights'][pid] = entry
    _read_inference_weights(temp, manifest)
    manifest['directories'] = sorted(str(p.relative_to(temp)) for p in temp.rglob('*') if p.is_dir())
    manifest['files'] = _hashes(temp)
    if not manifest['files']:
        raise ValueError('empty Algorithm checkpoint')
    (temp / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2))
    # Flush all files before publication; failed partial siblings stay invalid.
    for file in temp.rglob('*'):
        if file.is_file():
            with file.open('rb') as stream:
                os.fsync(stream.fileno())
    # Exclusive reservation prevents two writers from publishing the same name.
    # An interrupted empty reservation has no manifest and is never valid.
    path.mkdir()
    temp.rename(path)
    with _directory_fd(path.parent) as fd:
        os.fsync(fd)
    return manifest


def validate_bundle(path, expected, *, allow_runtime_changes=False):
    path = Path(path)
    if '.partial_' in path.name or path.is_symlink():
        raise ValueError('unpublished partial checkpoint')
    try:
        manifest = json.loads((path / 'manifest.json').read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError('missing or invalid checkpoint manifest') from exc
    if manifest.get('schema') != SCHEMA:
        raise ValueError('checkpoint schema mismatch: sep_pair_v2 numeric exports required; v1 is not converted')
    compatible(manifest['contract'], expected, allow_runtime_changes)
    for key in ('family', 'pv_encoder', 'trainable_ids', 'policy_ids'):
        if manifest[key] != expected[key]:
            raise ValueError(f'checkpoint {key} mismatch')
    if set(manifest['role_states']) != set(POLICY_IDS):
        raise ValueError('missing policy pair')
    for key in ('num_env_steps', 'iteration'):
        if type(manifest[key]) is not int or manifest[key] < 0:
            raise ValueError('invalid checkpoint counters')
    if (manifest.get('label') not in ('candidate', 'final', 'incomplete', 'tiny')
            or manifest.get('budget_reached') != (manifest['num_env_steps'] >= expected['budget'])
            or not manifest.get('joint_checkpoint_id')
            or set(manifest.get('initial_role_fingerprints', {})) != set(POLICY_IDS)):
        raise ValueError('invalid checkpoint metadata')
    if manifest['label'] == 'final' and (not manifest['budget_reached'] or expected['profile'] != 'main'):
        raise ValueError('invalid final checkpoint claim')
    if manifest['model_space_fingerprint'] != fingerprint({'models': expected['models'], 'spaces': expected['spaces']}):
        raise ValueError('model/space mismatch')
    directories = sorted(str(p.relative_to(path)) for p in path.rglob('*') if p.is_dir())
    if not manifest['files'] or _hashes(path) != manifest['files'] or directories != manifest['directories']:
        raise ValueError('corrupt, mixed or orphan checkpoint files')
    if not (path / 'algorithm' / 'algorithm_state.pkl').is_file():
        raise ValueError('missing whole Algorithm state')
    _read_inference_weights(path, manifest)
    return manifest


def restore_bundle(algorithm, path, expected, *, allow_runtime_changes=False):
    manifest = validate_bundle(path, expected, allow_runtime_changes=allow_runtime_changes)
    algorithm.restore(str(Path(path).resolve() / 'algorithm'))
    if _states(algorithm, expected) != manifest['role_states']:
        raise ValueError('restored policy/optimizer state mismatch')
    counters = algorithm._counters
    if (counters['num_env_steps_sampled'] != manifest['num_env_steps']
            or algorithm.iteration != manifest['iteration']):
        raise ValueError('restored Algorithm counters mismatch')
    return manifest
