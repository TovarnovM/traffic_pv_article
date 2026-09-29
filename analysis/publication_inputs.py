"""Strict loader for the published scalar projection (no simulator imports)."""
import gzip
import json
from itertools import product
from pathlib import Path

import numpy as np
import publication as main

ROAD_FIELDS = ('scenario_key', 'physical_state_hash', 'pre_av_tag_state_hash',
               'priority_vehicle_id', 'measurement_rng_seed', 'warmup_rng_seed')
FIELDS = ('method', 'training_seed', 'density_token', 'fraction_token', 'run_index',
          'joint_checkpoint_id', *ROAD_FIELDS, *main.METRICS, main.LAP,
          'completed_priority_laps', 'flow_priority', 'priority_vehicle_count',
          'vehicle_count', 'measurement_steps', 'status', 'split', 'protocol_hash')
REVIEW = Path(__file__).resolve().parents[1]


def validate(rows, winners):
    """Expand B1 aliases only after validating physical coverage and scalar schema."""
    if len(rows) != 29700:
        raise ValueError('physical row coverage')
    for row in rows:
        if set(row) != set(FIELDS):
            raise ValueError('row fields differ from projection schema')
        if row['method'] == 'B1' and row['fraction_token'] != 90:
            raise ValueError('B1 must have one physical row per road')
    expanded = [dict(r, fraction_token=f) for r in rows
                for f in (main.FRACTIONS if r['method'] == 'B1' else (r['fraction_token'],))]
    indexed = main.index_rows(expanded)
    selected = {(r['family'], r['seed']): r for r in winners}
    if len(winners) != 15 or set(selected) != set(product(main.FAMILIES, main.SEEDS)):
        raise ValueError('selection coverage/duplicates')
    ids = [r['candidate']['joint_checkpoint_id'] for r in winners]
    if len(set(ids)) != 15:
        raise ValueError('duplicate selected model')
    for r in expanded:
        base = indexed['B1', None, r['density_token'], 90, r['run_index']]
        if any(r[k] != base[k] for k in ROAD_FIELDS):
            raise ValueError('shared road identity mismatch')
        if r['method'] in main.FAMILIES and r['joint_checkpoint_id'] != selected[r['method'], r['training_seed']]['candidate']['joint_checkpoint_id']:
            raise ValueError('raw model differs from selected model')
        for metric in (*main.METRICS, 'flow_priority'):
            if r[metric] is None or not np.isfinite(r[metric]):
                raise ValueError(f'invalid complete metric: {metric}')
        lap = r[main.LAP]
        if (lap is None) != (r['completed_priority_laps'] == 0) or (lap is not None and (not np.isfinite(lap) or lap <= 0)):
            raise ValueError('lap completion mismatch')
        if r['completed_priority_laps'] < 0 or r['priority_vehicle_count'] != 1 or r['measurement_steps'] != 2000 or r['vehicle_count'] != r['density_token'] * 40 or r['status'] != 'ok' or r['split'] != 'test':
            raise ValueError('measurement metadata mismatch')
        if max(abs(r['flow_all'] - r['flow_background'] - r['mean_speed_priority']/4000),
               abs(r['flow_priority'] - r['mean_speed_priority']/4000)) > 1e-12:
            raise ValueError('flow identity fails')
    return indexed


def load_inputs(root=REVIEW):
    reference = json.loads((root / 'provenance/publication_reference.json').read_text())
    for name, digest in reference['input_sha256'].items():
        if main.sha(root / name) != digest:
            raise ValueError(f'published input hash changed: {name}')
    with gzip.open(root / 'data/evaluation/physical_rows.json.gz', 'rt', encoding='utf-8') as stream:
        rows = json.load(stream)
    receipt = json.loads((root / 'data/selection/winners.json').read_text())
    metadata = json.loads((root / 'data/evaluation/metadata.json').read_text())
    return validate(rows, receipt['winners']), receipt, metadata, reference
