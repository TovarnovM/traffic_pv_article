"""Fast manifest/seed contract check; no simulation or model loading."""
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'code'
sys.dont_write_bytecode = True
assert sys.flags.isolated, 'Run with python -I'
sys.path[:0] = [str(CODE / 'src'), str(CODE)]
from sep_research import protocol


class Scenarios(unittest.TestCase):
    def test_published_grid_and_seeds(self):
        manifest = json.loads((ROOT / 'provenance/science_sources.json').read_text())
        path = Path(protocol.__file__).resolve()
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         manifest['sources'][str(path.relative_to(CODE))])
        rows = json.loads((ROOT / 'data/scenarios/seeds.json').read_text())
        expected = [r._asdict() for r in protocol.seed_manifest('test', 710009)]
        self.assertEqual(rows, expected)
        hashes = json.loads((ROOT / 'data/scenarios/hashes.json').read_text())
        roads = {r['scenario_key']: r for r in hashes['roads']}
        roles = {(r['scenario_key'], r['fraction_token']): r for r in hashes['role_assignments']}
        self.assertEqual(len(hashes['roads']), 540)
        self.assertEqual(len(roads), 540)
        self.assertEqual(len(hashes['role_assignments']), 1620)
        self.assertEqual(set(roles), {(r['scenario_key'], r['fraction_token']) for r in rows})
        self.assertEqual(set(roads), {r['scenario_key'] for r in rows})
        for row in rows:
            road = roads[row['scenario_key']]
            self.assertEqual((road['density_token'], road['run_index']),
                             (row['density_token'], row['run_index']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
