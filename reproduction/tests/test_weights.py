"""Synthetic stdlib-only weights tests; no original archives or numerical imports."""

import copy
import hashlib
import io
import json
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
import warnings
import zipfile
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from policies import load_state_dict, verify_weights
from policies import weights


def digest(data):
    return {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def zipped(entries):
    buffer = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in entries:
                archive.writestr(name, data)
    return buffer.getvalue()


def npy(count=1, descr='<f4', version=1, header=None):
    if header is None:
        header = repr({'descr': descr, 'fortran_order': False, 'shape': (count,)})
    start = 10 if version == 1 else 12
    raw = header.encode('ascii')
    raw += b' ' * ((-start - len(raw) - 1) % 64) + b'\n'
    return b'\x93NUMPY' + bytes((version, 0)) + len(raw).to_bytes(start - 8, 'little') + raw + bytes(count * 4)


def fixture():
    models, entries = [], []
    for family, seeds in weights.MATRIX.items():
        for seed, iteration in seeds.items():
            model = dict(family=family, seed=seed, iteration=iteration,
                         joint_checkpoint_id=f'{family}-{seed}-{iteration}', pv_encoder='test',
                         source_manifest_sha256='0' * 64, original_provenance_flags={}, env={}, policies={})
            for role in sorted(weights.ROLES[family]):
                count = (355460 if role == 'av_policy' else 23812 if family in {'P_L', 'J_L'}
                         else 411572 if family == 'J_NoMP' else 411012)
                payload = zipped([('weight.npy', npy(count))])
                member = f'{family}/{seed}/{role}.npz'
                model['policies'][role] = dict(member=member, **digest(payload), encoder='test',
                                              parameter_count=count,
                                              arrays={'weight': {'shape': [count], 'dtype': 'float32'}})
                entries.append((member, payload))
            models.append(model)
    archive = zipped(entries)
    index = dict(schema='selected_policy_weights_v1', version='1.1.0',
                 archive=dict(file='selected_policies.zip', **digest(archive)), models=models,
                 aliases={'D2_L': {'family': 'J_L', 'role': 'av_policy', 'pv': 'native'}})
    return index, archive, entries


class WeightsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index, cls.archive, cls.entries = fixture()

    def package(self, root, index=None, archive=None):
        (root / 'MODELS.json').write_text(json.dumps(self.index if index is None else index))
        (root / 'selected_policies.zip').write_bytes(self.archive if archive is None else archive)

    def test_complete_package_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.package(root)
            with mock.patch.dict(sys.modules, {'numpy': None, 'torch': None}):
                self.assertEqual(len(verify_weights(root)['models']), 21)
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/verify_weights.py'), tmp],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('31 policy states', result.stdout)
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})

    def test_index_rejections(self):
        mutations = [
            lambda x: x['models'].pop(),
            lambda x: x['models'].__setitem__(1, copy.deepcopy(x['models'][0])),
            lambda x: x['models'][0].__setitem__('iteration', 501),
            lambda x: x['models'][0].__setitem__('seed', True),
            lambda x: x['models'][0]['policies'].__setitem__('pv_policy', x['models'][0]['policies']['av_policy']),
            lambda x: x['models'][0]['policies']['av_policy'].__setitem__('member', '../bad.npz'),
            lambda x: x['models'][0]['policies']['av_policy'].__setitem__('parameter_count', 1),
            lambda x: x['models'][0]['policies']['av_policy']['arrays']['weight'].__setitem__('dtype', 'object'),
            lambda x: x['models'][0]['policies']['av_policy']['arrays']['weight'].__setitem__('shape', [True]),
            lambda x: x['models'][0]['env'].__setitem__('bad', []),
            lambda x: x['aliases']['D2_L'].__setitem__('pv', 'learned'),
            lambda x: x['archive'].__setitem__('file', '../other.zip'),
        ]
        for mutate in mutations:
            index = copy.deepcopy(self.index)
            mutate(index)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                weights._index(index)

    def test_duplicate_json_and_cli_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.package(root)
            (root / 'MODELS.json').write_text('{"schema": 1, "schema": 2}')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                verify_weights(root)
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/verify_weights.py'), tmp],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)

    def test_outer_zip_rejections(self):
        link = zipfile.ZipInfo('link.npz')
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        cases = [self.entries[:-1], self.entries + self.entries[:1],
                 self.entries + [('../bad.npz', b'')], self.entries + [('extra.npz', b'')],
                 self.entries + [('folder/', b'')], self.entries + [(link, b'target')],
                 self.entries + [('/absolute.npz', b'')], self.entries + [('a\\b.npz', b'')]]
        for entries in cases:
            archive = zipped(entries)
            index = copy.deepcopy(self.index)
            index['archive'].update(digest(archive))
            with tempfile.TemporaryDirectory() as tmp, self.subTest(entries=len(entries)):
                self.package(Path(tmp), index, archive)
                with self.assertRaises(ValueError):
                    verify_weights(tmp)

    def test_hash_and_size_rejections(self):
        for level in ('archive', 'policy'):
            for field, value in [('bytes', 1), ('sha256', 'f' * 64)]:
                index = copy.deepcopy(self.index)
                metadata = index['archive'] if level == 'archive' else index['models'][0]['policies']['av_policy']
                metadata[field] = value
                with tempfile.TemporaryDirectory() as tmp, self.subTest(level=level, field=field):
                    self.package(Path(tmp), index)
                    with self.assertRaises(ValueError):
                        verify_weights(tmp)

    def test_npy_versions_and_malformed_headers(self):
        spec = {'shape': [1], 'dtype': 'float32'}
        for version in (1, 2, 3):
            weights._npy(npy(version=version), spec)
        cases = [b'', b'not an array', npy(descr='|O'), npy(descr='>f4'), npy(count=2),
                 npy()[:-1], npy() + b'\0', npy(version=4),
                 npy(header="{'descr':'<f4','shape':(True,), 'fortran_order':False}"),
                 npy(header="{'descr':'<f4','shape':(1,), 'fortran_order':0}"),
                 npy(header="{'descr':'<f4','descr':'<f4','shape':(1,), 'fortran_order':False}"),
                 npy(header="__import__('os').getcwd()"),
                 b'\x93NUMPY\x01\x00\xff\xff' + b' ' * 65535]
        for payload in cases:
            with self.subTest(prefix=payload[:12]), self.assertRaises(ValueError):
                weights._npy(payload, spec)

    def test_inner_zip_rejections(self):
        good = ('weight.npy', npy())
        for entries in [[good, good], [('other.npy', npy())], [('../weight.npy', npy())],
                        [('weight.npy', npy(descr='|O'))], [('weight.npy', npy()[:-1])]]:
            payload = zipped(entries)
            metadata = dict(**digest(payload), arrays={'weight': {'shape': [1], 'dtype': 'float32'}})
            with self.subTest(entries=len(entries)), self.assertRaises(ValueError):
                weights._policy(payload, metadata)

    def test_reader_rejects_before_numpy_import(self):
        with mock.patch.dict(sys.modules, {'numpy': None, 'torch': None}):
            for family, seed, role in [('A', 20260914, 'pv_policy'), ('D2_L', 20260914, 'pv_policy'),
                                       ('P_L', 20260914, 'av_policy'), ('A', 0, 'av_policy')]:
                with self.subTest(family=family), self.assertRaises(ValueError):
                    load_state_dict('/nonexistent', family, seed, role)
            with tempfile.TemporaryDirectory() as tmp:
                self.package(Path(tmp), archive=b'corrupt')
                with self.assertRaisesRegex(ValueError, 'byte count'):
                    load_state_dict(tmp, 'A', 20260914, 'av_policy')


if __name__ == '__main__':
    unittest.main()
