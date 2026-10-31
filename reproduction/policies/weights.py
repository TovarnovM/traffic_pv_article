"""Read only the explicitly supplied public weights package; never extract files."""

import ast
import hashlib
import io
import json
import math
from pathlib import Path
import re
import stat
import zipfile
import zlib


OLD_SEEDS = (20260914, 20260915, 20260916)
NEW_SEEDS = (2026102804, 2026102805)
MATRIX = {
    'A': dict(zip(OLD_SEEDS + NEW_SEEDS, (500, 600, 450, 550, 450))),
    'P_L': dict(zip(OLD_SEEDS, (550, 500, 500))),
    'P_G': dict(zip(OLD_SEEDS, (500, 450, 450))),
    'J_L': dict(zip(OLD_SEEDS, (450, 550, 450))),
    'J_G': dict(zip(OLD_SEEDS + NEW_SEEDS, (450, 500, 450, 450, 450))),
    'J_NoMP': dict(zip(NEW_SEEDS, (450, 500))),
}
ROLES = {family: ({'av_policy'} if family == 'A' else {'pv_policy'}
                 if family in {'P_L', 'P_G'} else {'av_policy', 'pv_policy'})
         for family in MATRIX}
MAX_HEADER = 10000
MAX_NPZ = 4 * 1024 * 1024
MAX_ARCHIVE = 128 * 1024 * 1024


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def _safe_name(name):
    _require(isinstance(name, str) and name and '\\' not in name and ':' not in name
             and all(ord(c) >= 32 and ord(c) != 127 for c in name)
             and all(p not in {'', '.', '..'} for p in name.split('/')),
             'unsafe archive member name')


def _hash(value):
    _require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value),
             'invalid SHA-256')


def _size(value, maximum):
    _require(type(value) is int and 0 < value <= maximum, 'invalid byte count')


def _check_bytes(data, metadata):
    _require(len(data) == metadata['bytes'], 'byte count mismatch')
    _require(hashlib.sha256(data).hexdigest() == metadata['sha256'], 'SHA-256 mismatch')


def _members(archive, expected):
    found = {}
    for info in archive.infolist():
        _safe_name(info.filename)
        kind = stat.S_IFMT(info.external_attr >> 16)
        _require(not info.is_dir() and kind in {0, stat.S_IFREG}
                 and not (info.external_attr & 0x10), 'non-file ZIP member')
        _require(info.orig_filename == info.filename, 'invalid ZIP member name')
        _require(info.filename not in found, 'duplicate ZIP member')
        _require(not info.flag_bits & 1, 'encrypted ZIP member')
        found[info.filename] = info
    _require(set(found) == set(expected), 'unexpected or missing ZIP members')
    return found


def _npy(data, metadata):
    _require(data[:6] == b'\x93NUMPY', 'invalid NPY magic')
    version = data[6:8]
    _require(version in {b'\x01\x00', b'\x02\x00', b'\x03\x00'}, 'unsupported NPY version')
    start = 10 if version == b'\x01\x00' else 12
    _require(len(data) >= start, 'truncated NPY prefix')
    length = int.from_bytes(data[8:start], 'little')
    end = start + length
    _require(0 < length <= MAX_HEADER and end <= len(data), 'invalid NPY header length')
    raw = data[start:end]
    _require(raw.endswith(b'\n'), 'invalid NPY header termination')
    try:
        tree = ast.parse(raw.decode('utf-8' if version == b'\x03\x00' else 'latin1').strip(), mode='eval')
        _require(isinstance(tree.body, ast.Dict), 'invalid NPY header dictionary')
        keys = [ast.literal_eval(key) for key in tree.body.keys]
        _require(len(keys) == len(set(keys)), 'duplicate NPY header keys')
        header = ast.literal_eval(tree)
    except (SyntaxError, TypeError, UnicodeError, RecursionError) as exc:
        raise ValueError('invalid NPY header') from exc
    _require(set(header) == {'descr', 'fortran_order', 'shape'}, 'invalid NPY header fields')
    _require(header['descr'] == '<f4', 'NPY dtype must be little-endian float32')
    _require(type(header['fortran_order']) is bool, 'invalid NPY order')
    shape = header['shape']
    _require(type(shape) is tuple and all(type(n) is int and n >= 0 for n in shape)
             and list(shape) == metadata['shape'], 'NPY shape mismatch')
    _require(len(data) - end == math.prod(shape) * 4, 'NPY payload length mismatch')


def _policy(data, metadata):
    _check_bytes(data, metadata)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        expected = {key + '.npy': value for key, value in metadata['arrays'].items()}
        members = _members(archive, expected)
        for name, spec in expected.items():
            limit = math.prod(spec['shape']) * 4 + MAX_HEADER + 12
            _require(members[name].file_size <= limit, 'oversized NPY member')
            _npy(archive.read(members[name]), spec)


def _index(index):
    _require(index['schema'] == 'selected_policy_weights_v1' and index['version'] == '1.1.0',
             'unsupported weights index')
    archive = index['archive']
    _require(archive['file'] == 'selected_policies.zip', 'unexpected archive filename')
    _hash(archive['sha256'])
    _size(archive['bytes'], MAX_ARCHIVE)
    _require(index['aliases'] == {'D2_L': {'family': 'J_L', 'role': 'av_policy', 'pv': 'native'}},
             'invalid D2_L alias')
    models = index['models']
    _require(isinstance(models, list) and len(models) == 21, 'expected 21 models')
    seen, policies = set(), {}
    for model in models:
        family, seed, iteration = model['family'], model['seed'], model['iteration']
        _require(isinstance(family, str) and family in MATRIX and type(seed) is int
                 and type(iteration) is int and MATRIX[family].get(seed) == iteration,
                 'unexpected family/seed/iteration')
        _require((family, seed) not in seen, 'duplicate model key')
        seen.add((family, seed))
        _hash(model['source_manifest_sha256'])
        _require(isinstance(model['joint_checkpoint_id'], str) and model['joint_checkpoint_id']
                 and isinstance(model['pv_encoder'], str)
                 and isinstance(model['original_provenance_flags'], dict), 'invalid model provenance')
        _require(isinstance(model['env'], dict) and all(
            value is None or type(value) in {str, bool, int} or
            (type(value) is float and math.isfinite(value)) for value in model['env'].values()),
            'environment constants must be finite JSON scalars')
        _require(set(model['policies']) == ROLES[family], 'inconsistent policy roles')
        for role, policy in model['policies'].items():
            member = policy['member']
            _safe_name(member)
            _require(member == f'{family}/{seed}/{role}.npz' and member not in policies,
                     'inconsistent or duplicate policy member')
            _hash(policy['sha256'])
            _size(policy['bytes'], MAX_NPZ)
            _require(isinstance(policy['encoder'], str) and policy['encoder'], 'invalid encoder')
            if role == 'pv_policy':
                _require(policy['encoder'] == model['pv_encoder'], 'inconsistent PV encoder')
            arrays = policy['arrays']
            _require(isinstance(arrays, dict) and arrays, 'missing array metadata')
            count = 0
            for key, spec in arrays.items():
                _safe_name(key)
                _require('/' not in key and isinstance(spec['shape'], list)
                         and all(type(n) is int and n >= 0 for n in spec['shape'])
                         and spec['dtype'] == 'float32', 'invalid array metadata')
                count += math.prod(spec['shape'])
            expected = (355460 if role == 'av_policy' else 23812 if family in {'P_L', 'J_L'}
                        else 411572 if family == 'J_NoMP' else 411012)
            _require(type(policy['parameter_count']) is int
                     and policy['parameter_count'] == count == expected, 'parameter count mismatch')
            policies[member] = policy
    _require(seen == {(family, seed) for family, seeds in MATRIX.items() for seed in seeds}
             and len(policies) == 31, 'expected complete 21-model/31-policy selection')
    return policies


def _verified(weights_dir, requested=None):
    root = Path(weights_dir)
    try:
        with (root / 'MODELS.json').open('rb') as stream:
            raw = stream.read(1024 * 1024 + 1)
        _require(len(raw) <= 1024 * 1024, 'weights index too large')
        index = json.loads(raw, object_pairs_hook=_unique_object)
        policies = _index(index)
        selected = None
        with (root / index['archive']['file']).open('rb') as stream:
            data = stream.read(MAX_ARCHIVE + 1)
        _check_bytes(data, index['archive'])
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = _members(archive, policies)
            for name, policy in policies.items():
                _require(members[name].file_size == policy['bytes'], 'NPZ byte count mismatch')
                payload = archive.read(members[name])
                _policy(payload, policy)
                if name == requested:
                    selected = payload
        return index, selected
    except (OSError, KeyError, TypeError, AttributeError, UnicodeError, RecursionError,
            zipfile.BadZipFile, NotImplementedError, RuntimeError, EOFError, zlib.error) as exc:
        raise ValueError(f'invalid weights package ({type(exc).__name__})') from exc


def verify_weights(weights_dir):
    """Verify the fixed 21-model/31-policy package; return its public index."""
    return _verified(weights_dir)[0]


def load_state_dict(weights_dir, family, seed, role):
    """Return NumPy arrays, not a framework model; import NumPy only after validation."""
    if family == 'D2_L':
        if role != 'av_policy':
            raise ValueError('D2_L uses native PV; no PV state dict exists')
        family = 'J_L'
    if family not in ROLES or role not in ROLES[family]:
        raise ValueError('unknown family or missing/native policy role')
    if type(seed) is not int or seed not in MATRIX[family]:
        raise ValueError('unknown selected seed')
    _, payload = _verified(weights_dir, f'{family}/{seed}/{role}.npz')
    if payload is None:
        raise ValueError('selected policy is missing')
    import numpy as np
    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}
