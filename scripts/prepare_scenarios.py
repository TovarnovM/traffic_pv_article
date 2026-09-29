"""Generate exactly one published initial road and its three AV assignments."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'code'
sys.dont_write_bytecode = True
assert sys.flags.isolated, 'Run with python -I'
sys.path[:0] = [str(CODE / 'src'), str(CODE)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--density-token', type=int, choices=range(5, 91, 5), required=True)
    parser.add_argument('--run-index', type=int, choices=range(30), required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'provenance/science_sources.json').read_text())
    for relative, digest in manifest['sources'].items():
        assert hashlib.sha256((CODE / relative).read_bytes()).hexdigest() == digest, relative
    import numpy as np
    from sep_research import evaluate, protocol
    for name, module in list(sys.modules.items()):
        if name.split('.')[0] in ('snfs_traffic', 'sep_research', 'research') and getattr(module, '__file__', None):
            path = Path(module.__file__).resolve()
            relative = str(path.relative_to(CODE))
            assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['sources'][relative]
    seeds = json.loads((ROOT / 'data/scenarios/seeds.json').read_text())
    rows = [protocol.SeedRow(**r) for r in seeds
            if r['density_token'] == args.density_token and r['run_index'] == args.run_index]
    assert len(rows) == 3
    physical = json.loads((ROOT / 'configs/evaluation.json').read_text())['physical']
    args.out_dir.mkdir(parents=True, exist_ok=False)
    warm, pre, pv, states, backend = evaluate.prepare_group(physical, rows)
    historical = json.loads((ROOT / 'data/scenarios/hashes.json').read_text())
    road = next(r for r in historical['roads'] if r['scenario_key'] == rows[0].scenario_key)
    actual = dict(physical_state_hash=evaluate._fingerprint(pre, pv)['physical_state_hash'],
                  warmed_state_hash=evaluate._fingerprint(warm, pv)['activated_state_hash'],
                  pre_av_tag_state_hash=evaluate._fingerprint(pre, pv)['activated_state_hash'],
                  pv_role_hash=evaluate._fingerprint(pre, pv)['pv_role_hash'], priority_vehicle_id=int(pv))
    assert all(actual[k] == road[k] for k in actual), (actual, road)
    arrays = {f'{stage}_{name}': value for stage, state in [('warm', warm), ('pre', pre)]
              for name, value in evaluate.arrays(state).items()}
    assignments = []
    for fraction, (state, av) in sorted(states.items()):
        expected = next(r for r in historical['role_assignments']
                        if r['scenario_key'] == rows[0].scenario_key and r['fraction_token'] == fraction)
        fingerprint = evaluate._fingerprint(state, pv, av)
        assert all(fingerprint[k] == expected[k] for k in
                   ('activated_state_hash', 'role_state_hash', 'av_subset_hash'))
        arrays.update({f'av{fraction}_{name}': value for name, value in evaluate.arrays(state).items()})
        arrays[f'av{fraction}_vehicle_ids'] = np.asarray(av, dtype=np.int64)
        assignments.append(expected)
    np.savez_compressed(args.out_dir / 'initial_states.npz', **arrays)
    report = dict(status='PASS', road=road, role_assignments=assignments, backend=backend,
                  physical=physical, seeds=[r._asdict() for r in rows], warmup_steps=physical['warmup_steps'])
    (args.out_dir / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', scenarios=1, role_assignments=3)))


if __name__ == '__main__':
    main()
