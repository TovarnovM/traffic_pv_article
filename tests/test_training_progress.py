"""Public regression tests: only shipped training data and analysis are required."""
import copy
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

REVIEW = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('training_progress', REVIEW / 'analysis/training_progress.py')
tp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tp)


class TrainingProgressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.manifest = tp.load_data()

    def test_all_fifteen_histories_and_selection(self):
        grouped = tp.validate(self.rows, self.manifest)
        self.assertEqual(len(grouped), 15)
        for (family, seed), rows in grouped.items():
            self.assertEqual([r['iteration'] for r in rows], list(range(1, 612)))
            self.assertEqual(rows[-1]['env_steps'], 10010624)
            self.assertEqual([r['iteration'] for r in rows if r['selected_checkpoint']],
                             [tp.SELECTED[family][tp.SEEDS.index(seed)]])
        self.assertEqual(sum(len(l['segments']) for l in self.manifest['lines']), 34)

    def test_special_lineages_and_excluded_tails(self):
        grouped = tp.validate(self.rows, self.manifest)
        pg = grouped['P_G', 20260914]
        self.assertEqual([pg[i - 1]['source_row'] for i in [306, 307, 575, 576, 611]], [306, 1, 269, 1, 36])
        self.assertNotEqual(pg[574]['source_id'], pg[575]['source_id'])
        pl = grouped['P_L', 20260916]
        self.assertTrue(self.manifest['sources'][pl[0]['source_id']]['path'].endswith('attempt_003/source/metrics.jsonl'))
        rejected = self.manifest['rejected']
        tails = [r for r in rejected if 'tail-discarded' in r['path']]
        self.assertEqual(len(tails), 14)
        self.assertEqual([r['rows'] for r in tails], [12, 3, 9, 1, 3, 12, 3, 9, 5, 1, 4, 8, 11, 1])
        self.assertEqual(next(r for r in rejected if '/partial/' in r['path'])['rows'], 11)
        self.assertEqual(next(r for r in rejected if 'recovery_PL' in r['path'])['rows'], 105)
        rollback = next(r for r in tails if '/20260915.gen6/' in r['path'])
        self.assertEqual((rollback['first_iteration'], rollback['last_iteration']), (151, 154))
        generations = {l['seed']: l['final_resume_generation'] for l in self.manifest['lines'] if l['family'] == 'J_G'}
        self.assertEqual(generations, {20260914: 5, 20260915: 6, 20260916: 3})

    def test_negative_rows(self):
        for field, value in [('env_steps', 0), ('source_row', 0), ('source_id', -1),
                             ('segment', 2), ('selected_checkpoint', True),
                             ('pv_speed_mean', None), ('team_return_mean', float('nan')),
                             ('iteration', True)]:
            with self.subTest(field=field):
                rows = [dict(r) for r in self.rows]
                rows[0][field] = value
                with self.assertRaises(ValueError):
                    tp.validate(rows, self.manifest)
        with self.assertRaises(ValueError):
            tp.validate(self.rows[1:], self.manifest)
        rows = [dict(r) for r in self.rows]
        rows[1] = dict(rows[0])
        with self.assertRaises(ValueError):
            tp.validate(rows, self.manifest)

    def test_negative_manifest(self):
        for key, value in [('sha256', 'not-a-hash'), ('path', '/home/private/log.jsonl'),
                           ('path', '../raw.jsonl'), ('path', 'cpu/P_L-20260916/attempt_001/metrics.jsonl')]:
            with self.subTest(key=key, value=value):
                manifest = copy.deepcopy(self.manifest)
                manifest['sources'][0][key] = value
                with self.assertRaises(ValueError):
                    tp.validate(self.rows, manifest)
        manifest = copy.deepcopy(self.manifest)
        manifest['lines'][0]['segments'][0]['last_iteration'] = 305
        with self.assertRaises(ValueError):
            tp.validate(self.rows, manifest)

    def test_float_roundtrip_and_deterministic_gzip(self):
        original = (REVIEW / 'data/training/progress.jsonl.gz').read_bytes()
        self.assertEqual(original[4:8], b'\x00' * 4)
        payload = b''.join((json.dumps(r, separators=(',', ':'), allow_nan=False) + '\n').encode() for r in self.rows)
        self.assertEqual(payload, gzip.decompress(original))
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode='wb', filename='', mtime=0, compresslevel=9) as stream:
            stream.write(payload)
        self.assertEqual(buffer.getvalue(), original)

    def test_checksum_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory)
            shutil.copy(REVIEW / 'data/training/manifest.json', dest)
            data = bytearray((REVIEW / 'data/training/progress.jsonl.gz').read_bytes())
            data[-9] ^= 1
            (dest / 'progress.jsonl.gz').write_bytes(data)
            with self.assertRaisesRegex(ValueError, 'checksum'):
                tp.load_data(dest)

    def test_plot_never_connects_segments(self):
        import matplotlib.pyplot as plt
        figures = []
        def capture(fig):
            if hasattr(fig, 'axes'):
                figures.append(fig)
        with tempfile.TemporaryDirectory() as directory, patch.object(plt, 'close', side_effect=capture), patch('matplotlib.figure.Figure.savefig'):
            tp.plot(self.rows, self.manifest, directory)
        self.assertEqual(len(figures), 2)
        for fig in figures:
            for ax, family in zip(fig.axes, tp.FAMILIES):
                curves = [line for line in ax.lines if len(line.get_xdata()) > 2]
                ranges = [pair for seed in tp.SEEDS for pair in tp.expected_ranges(family, seed)]
                self.assertEqual(len(curves), len(ranges))
                for curve, (a, b) in zip(curves, ranges):
                    self.assertEqual(len(curve.get_xdata()), b - a + 1)
                    self.assertAlmostEqual(curve.get_xdata()[0], a * 16384 / 1e6)
                    self.assertAlmostEqual(curve.get_xdata()[-1], b * 16384 / 1e6)
            plt.close(fig)

    def test_isolated_cli_and_deterministic_figure_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / 'review'
            (package / 'analysis').mkdir(parents=True)
            shutil.copy(REVIEW / 'analysis/training_progress.py', package / 'analysis')
            shutil.copytree(REVIEW / 'data/training', package / 'data/training')
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(root / 'mpl'))
            for output in ['first', 'second']:
                result = subprocess.run([sys.executable, str(package / 'analysis/training_progress.py'),
                                         '--output-dir', str(root / output)], cwd=root, env=env,
                                        capture_output=True, text=True, check=True)
                self.assertIn('9165', result.stdout)
            first = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / 'first').iterdir()}
            second = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / 'second').iterdir()}
            self.assertEqual(len(first), 6)
            self.assertEqual(first, second)


if __name__ == '__main__':
    unittest.main()
