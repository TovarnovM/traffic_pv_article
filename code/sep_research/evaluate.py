"""Grouped, paired CPU evaluation. No Ray imports or pickle model loading.

Public API: evaluate(plan, out_dir, workers=1, resume=False) -> raw row list.
Read-only: read_evaluation(out_dir, expected_prepared=prepare_plan(expected_plan)).
CLI: python -m sep_research.evaluate --plan PLAN.json --out-dir DIR [--workers 2]

Plan JSON (paths relative to invocation cwd):
  {"purpose":"development", "seed_manifest":[... exact SeedRow._asdict() ...],
   "physical":{"road_length":100,"warmup_steps":2,"measure_steps":8,
               "backend":"reference"},
   "fixed_modes":["B1","B2","N","L"],
   "bundles":[{"family":"J_L","training_seed":810017,
               "bundle":".../checkpoints/tiny_1_16","run_config":".../run_config.json"}],
   "diagnostics":["D1_L","D2_L","D3_L","D4_L"], "legacy_cache":null}

seed_manifest may instead be an explicit JSON path containing that row list.
Omitted physical fields resolve to the article's DEFAULT_ENV_CONFIG; per-case
fields, family, encoder and training seed cannot be supplied there. Development
permits only road_length=100, warmup=0/2/5 and horizon=8/16 deviations (plus the
reference backend). Resolved deviations are recorded. Validation requires the
entire preregistered validation manifest, full physics, and all three training
seeds for every requested learned family. One bundle per family/seed per call;
validation of different epochs uses separate invocations. Test requires an exact
main/diagnostics plan from build_test_plan(selection_receipt_path, panel=...). No scientific claims
or selection are produced here. No persistent warm cache: one warm per group
per cold invocation, or zero for a verified complete resumed chunk.

Outputs: provenance.json, complete atomic chunks, raw.json/raw.csv, counts.json.
Failed groups produce failed_*.json and abort before matrix analysis. Resume
requires identical plan, source, runtime, seed and model hashes. Partial chunks
are never reused. Hashes check trusted-local integrity, not authenticity.
"""
import argparse
import csv
import io
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields
import json
import fcntl
import multiprocessing
import os
from pathlib import Path
import platform
import re
import tempfile

import numpy as np
import torch

from research import evaluate_pv_graph_ppo as graph_eval
from sep_research import protocol
from sep_research.env_core import SepResearchCore
from sep_research.inference import load_pair, predict
from sep_research.export_legacy import (ROOT, json_hash, sha256, write_json,
                                       load_legacy, legacy_prediction)
from snfs_traffic.core.roles import high_speed_vehicle_mask
from snfs_traffic.metrics import PriorityMetricsAccumulator
from snfs_traffic.rules import PriorityYieldConfig, PriorityYieldController
from snfs_traffic.scenarios import (AV_VEH_TYPE, PRIORITY_VEH_TYPE,
    PriorityVehicleConfig, assign_priority_vehicles, select_yielding_vehicle_ids)
from snfs_traffic.scenarios.priority import PRIORITY_BEHAVIOR_ID, PRIORITY_YIELD_BEHAVIOR_ID
from snfs_traffic.scenarios.init import HDV_BEHAVIOR_ID

_CASE_FIELDS = {'density', 'av_fraction', 'episode_steps'}
PHYSICAL_DEFAULTS = {k: v for k, v in graph_eval.DEFAULT_ENV_CONFIG.items() if k not in _CASE_FIELDS}
PHYSICAL_DEFAULTS['measure_steps'] = 2000
_BINDING_FIELDS = set(PHYSICAL_DEFAULTS) - {'warmup_steps', 'measure_steps', 'backend', 'validate'}
_MODELS = {}
_LEGACY = {}


def _positive(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def _test_inputs(selection_receipt_path, panel, *, expected_plan=None):
    # Lazy import: selection prepares validation plans through this module.
    from sep_research import admission
    if panel not in ('main', 'diagnostics'):
        raise ValueError('test panel must be main or diagnostics')
    profile = admission.load_frozen_profile()
    path = Path(selection_receipt_path)
    receipt = _read_json(path)
    digest = sha256(path)
    selected = admission.validate_selection(receipt, frozen_profile=profile)
    if sha256(path) != digest or json_hash(_read_json(path)) != json_hash(selected):
        raise ValueError('selection changed during verification')
    if expected_plan is not None and any(expected_plan.get(k) != v for k, v in dict(
            selection_receipt=str(path.resolve()), selection_sha256=digest,
            profile_sha256=json_hash(profile)).items()):
        raise ValueError('test selection/profile binding mismatch')
    physical = profile['evaluation_physical']
    if physical != PHYSICAL_DEFAULTS or physical['warmup_steps'] != 1000 or physical['measure_steps'] != 2000:
        raise ValueError('anchored test physical/horizon mismatch')
    versions = dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__)
    if versions != {k: profile['runtime']['versions'][k] for k in versions}:
        raise ValueError('anchored test runtime mismatch')
    fixed, legacy = profile['fixed_controls'], profile['legacy_cache']
    if set(fixed) != set(protocol.FIXED_CONTROLS):
        raise ValueError('anchored fixed controls mismatch')
    for method in protocol.FIXED_CONTROLS:
        identity = (json_hash(legacy['manifest']['models'][method]) if method.endswith('_old') else
                    json_hash(dict(method=method, config=physical, rule_detection=30, rule_cooldown=1)))
        if fixed[method] != identity:
            raise ValueError('anchored fixed identity mismatch')
    if panel == 'main' and (str(Path(legacy['path']).resolve()) != legacy['path']
            or sha256(Path(legacy['path']) / 'manifest.json') != legacy['manifest_sha256']
            or _read_json(Path(legacy['path']) / 'manifest.json') != legacy['manifest']):
        raise ValueError('anchored legacy cache mismatch')
    winners = {(w['family'], w['seed']): w['candidate'] for w in selected['winners']}
    origins = {(o['family'], o['seed']): o for o in selected['origins']}
    pairs = [(f, s) for f in protocol.FAMILIES for s in protocol.TRAINING_SEEDS]
    if len(selected['winners']) != 15 or set(winners) != set(pairs) or set(origins) != set(pairs):
        raise ValueError('test requires all 15 selected pairs')
    bundles = []
    for family, seed in pairs:
        if panel == 'diagnostics' and family not in ('J_L', 'J_G'):
            continue
        candidate, origin = winners[(family, seed)], origins[(family, seed)]
        bundle, run = Path(candidate['path']), Path(origin['run_dir']) / 'run_config.json'
        manifest = _read_json(bundle / 'manifest.json')
        if (str(bundle.resolve()) != candidate['path']
                or candidate not in origin['candidates']
                or sha256(bundle / 'manifest.json') != candidate['manifest_sha256']
                or sha256(run) != origin['files']['run_config.json']
                or _read_json(run)['contract'] != profile['contracts'][f'{family}:{seed}']
                or any(manifest[k] != candidate[k] for k in
                       ('joint_checkpoint_id', 'role_states', 'initial_role_fingerprints'))):
            raise ValueError('selected pair/origin changed before test preparation')
        bundles.append(dict(family=family, training_seed=seed, bundle=candidate['path'],
                            run_config=str(Path(origin['run_dir']) / 'run_config.json')))
    # Only now expand requested cases. Training exclusion may already have read
    # the reserved registry during frozen-profile verification.
    manifest = protocol.seed_manifest('test', protocol.TEST_ROOT,
        density_tokens=protocol.DENSITY_TOKENS if panel == 'main' else
                       tuple(protocol.DENSITY_TOKENS[i] for i in protocol.DIAGNOSTIC_DENSITY_IDS))
    plan = dict(purpose='test', panel=panel, selection_receipt=str(path.resolve()),
        selection_sha256=digest, profile_sha256=json_hash(profile),
        seed_manifest=[r._asdict() for r in manifest], physical=physical,
        fixed_modes=list(protocol.FIXED_CONTROLS) if panel == 'main' else [], bundles=bundles,
        diagnostics=[] if panel == 'main' else list(protocol.DIAGNOSTICS),
        legacy_cache=legacy['path'] if panel == 'main' else None)
    return plan, profile, selected


def build_test_plan(selection_receipt_path, *, panel):
    """Build an exact panel after full selection verification; never execute.

    Receipt location and byte digest, plus the designated profile digest, bind
    the plan. prepare_plan rechecks them; no cached authorization.
    """
    return _test_inputs(selection_receipt_path, panel)[0]


def _test_matrix(plan, profile, selected):
    winners = {(w['family'], w['seed']): w['candidate'] for w in selected['winners']}
    instances = [(f, s) for f in (protocol.FAMILIES if plan['panel'] == 'main' else protocol.DIAGNOSTICS)
                 for s in protocol.TRAINING_SEEDS]
    matrix = []
    for method, seed in [(f, None) for f in plan['fixed_modes']] + instances:
        family, pv, av = composition(method)
        candidate = None if seed is None else winners[(family, seed)]
        matrix.append(dict(method=method, training_seed=seed, family=family, pv_mode=pv, av_mode=av,
            joint_checkpoint_id=profile['fixed_controls'][method] if candidate is None else candidate['joint_checkpoint_id']))
    logical, physical = (35640, 34560) if plan['panel'] == 'main' else (8640, 8640)
    return dict(schema='sep_heldout_gate_v1', panel=plan['panel'],
        selection_receipt=plan['selection_receipt'], selection_sha256=plan['selection_sha256'],
        profile_sha256=plan['profile_sha256'], matrix=matrix,
        logical_rows=logical, physical_trajectories=physical, measurement_steps=physical * 2000)


def _check_test_prepared(prepared, plan, profile, selected):
    proof = _test_matrix(plan, profile, selected)
    provenance = prepared['provenance']
    checkpoints = {(r['method'], r['training_seed']): r['joint_checkpoint_id'] for r in proof['matrix']}
    if (json_hash(prepared['plan']) != json_hash(plan) or provenance.get('heldout_gate') != proof
            or prepared['checkpoints'] != checkpoints
            or provenance['runtime'] != {k: profile['runtime']['versions'][k] for k in ('python', 'numpy', 'torch')}
            or any(profile['sources'].get(k) != v for k, v in provenance['sources'].items())
            or prepared['protocol_hash'] != profile['protocol_sha256']):
        raise ValueError('heldout preparation/anchor mismatch')
    if set(provenance['models']) != {f"{b['family']}:{b['training_seed']}" for b in plan['bundles']}:
        raise ValueError('heldout model matrix mismatch')
    for item in plan['bundles']:
        key = (item['family'], item['training_seed'])
        candidate = next(w['candidate'] for w in selected['winners'] if (w['family'], w['seed']) == key)
        origin = next(o for o in selected['origins'] if (o['family'], o['seed']) == key)
        model = provenance['models'][f'{key[0]}:{key[1]}']
        if (model['manifest_sha256'] != candidate['manifest_sha256']
                or model['run_config_sha256'] != origin['files']['run_config.json']
                or any(model['manifest'][k] != candidate[k] for k in
                       ('joint_checkpoint_id', 'role_states', 'initial_role_fingerprints'))):
            raise ValueError('heldout selected pair/origin mismatch')
    if provenance['legacy'] != (profile['legacy_cache']['manifest'] if plan['panel'] == 'main' else None):
        raise ValueError('heldout fixed cache mismatch')


def prepare_plan(plan):
    """Validate all inputs, identities and numeric models before any simulation."""
    test_inputs = None
    if plan.get('purpose') == 'test':
        if not {'panel', 'selection_receipt', 'selection_sha256', 'profile_sha256'} <= set(plan):
            raise ValueError('test requires selection receipt and panel binding')
        test_inputs = _test_inputs(plan['selection_receipt'], plan['panel'], expected_plan=plan)
        if json_hash(plan) != json_hash(test_inputs[0]):
            raise ValueError('test requires exact anchored panel plan')
    if plan.get('purpose') not in ('development', 'validation', 'test'):
        raise ValueError('purpose must be development, validation or test')
    allowed = {'purpose', 'seed_manifest', 'physical', 'fixed_modes', 'bundles', 'diagnostics', 'legacy_cache'}
    if test_inputs:
        allowed |= {'panel', 'selection_receipt', 'selection_sha256', 'profile_sha256'}
    if set(plan) - allowed or not {'purpose', 'seed_manifest', 'physical', 'fixed_modes', 'bundles', 'diagnostics'} <= set(plan):
        raise ValueError('missing or unknown plan fields')
    plan = json.loads(json.dumps(plan, allow_nan=False))
    seed_input = plan['seed_manifest']
    if isinstance(seed_input, str):
        seed_input = json.loads(Path(seed_input).read_text())
    manifest = tuple(protocol.SeedRow(**r) for r in seed_input)
    protocol.validate_seed_manifest(manifest)
    purpose = plan['purpose']
    root = {'development': protocol.DEVELOPMENT_ROOT, 'validation': protocol.VALIDATION_ROOT,
            'test': protocol.TEST_ROOT}[purpose]
    if any(r.split != purpose or r.split_root != root for r in manifest):
        raise ValueError('purpose/split or immutable manifest root mismatch')
    # Require a rectangular explicit manifest, including the same fractions/runs in each density.
    expected = protocol.seed_manifest(purpose, root,
        density_tokens=sorted({r.density_token for r in manifest}),
        fraction_tokens=sorted({r.fraction_token for r in manifest}),
        runs=max(r.run_index for r in manifest) + 1)
    if set(manifest) != set(expected):
        raise ValueError('missing manifest conditions/runs')
    if purpose == 'validation' and set(manifest) != set(protocol.seed_manifest(
            purpose, root, density_tokens=(30,), fraction_tokens=(100,), runs=20)):
        raise ValueError('validation requires the exact preregistered 20-run manifest')
    if set(plan['physical']) - set(PHYSICAL_DEFAULTS):
        raise ValueError('unknown or per-case physical config field')
    physical = dict(PHYSICAL_DEFAULTS, **plan['physical'])
    deviations = {k: {'article': v, 'actual': physical[k]} for k, v in PHYSICAL_DEFAULTS.items() if physical[k] != v}
    permitted = {'road_length', 'warmup_steps', 'measure_steps', 'backend', 'validate'} if purpose == 'development' else {'validate'}
    if set(deviations) - permitted:
        raise ValueError('physical/sensor/control settings must match article')
    _positive(physical['measure_steps'], 'measure_steps')
    _positive(physical['warmup_steps'], 'warmup_steps', 0)
    if purpose == 'development' and deviations:
        if (physical['road_length'] not in (100, 1000) or physical['warmup_steps'] not in (0, 2, 5, 1000)
                or physical['measure_steps'] not in (8, 16, 2000)
                or physical['backend'] not in ('reference', 'optimized')):
            raise ValueError('unsupported explicit development deviation')
    fixed, diagnostics = plan['fixed_modes'], plan['diagnostics']
    if len(set(fixed)) != len(fixed) or not set(fixed) <= set(protocol.FIXED_CONTROLS):
        raise ValueError('duplicate or unknown fixed modes')
    if len(set(diagnostics)) != len(diagnostics) or not set(diagnostics) <= set(protocol.DIAGNOSTICS):
        raise ValueError('duplicate or unknown diagnostics')
    checkpoints, bundles, model_manifest = {}, [], {}
    common_sources = None
    for item in plan['bundles']:
        if set(item) != {'family', 'training_seed', 'bundle', 'run_config'}:
            raise ValueError('explicit bundle, run_config, family and training_seed required; no role pairing')
        family, seed = item['family'], item['training_seed']
        _positive(seed, 'training_seed', 0)
        if family not in protocol.FAMILIES or (family, seed) in checkpoints:
            raise ValueError('unknown family or duplicate bundle instance')
        if purpose != 'development' and seed not in protocol.TRAINING_SEEDS:
            raise ValueError('invalid preregistered training seed')
        path, run_path = Path(item['bundle']).resolve(), Path(item['run_config']).resolve()
        contract = json.loads(run_path.read_text())['contract']
        encoder = 'graph_mlp' if family.endswith('_G') else 'local_mlp'
        trainable = ['av_policy'] if family == 'A' else ['pv_policy'] if family.startswith('P_') else ['pv_policy', 'av_policy']
        if (contract['family'] != family or contract['seed'] != seed or contract['pv_encoder'] != encoder
                or contract['env']['controller_mode'] != family or contract['env']['pv_encoder'] != encoder
                or contract['trainable_ids'] != trainable):
            raise ValueError('bundle family/seed/role/encoder binding mismatch')
        if purpose != 'development' and contract['profile'] != 'main':
            raise ValueError('validation/test requires main-profile bundles')
        for key in _BINDING_FIELDS:
            if contract['env'].get(key, PHYSICAL_DEFAULTS[key]) != physical[key]:
                if not (purpose == 'development' and key == 'road_length'):
                    raise ValueError(f'checkpoint physical mismatch: {key}')
        sources = contract['sources']
        if 'sep_research/env_core.py' not in sources:
            raise ValueError('missing core source identity')
        for name, digest in sources.items():
            source = (ROOT / name).resolve()
            if not source.is_relative_to(ROOT) or sha256(source) != digest:
                raise ValueError(f'checkpoint source identity mismatch: {name}')
        if common_sources is not None and sources != common_sources:
            raise ValueError('mixed checkpoint sources')
        common_sources = sources
        _, receipt = load_pair(path, contract)
        checkpoint = receipt['joint_checkpoint_id']
        if checkpoint in checkpoints.values():
            raise ValueError('duplicate pair ID')
        checkpoints[(family, seed)] = checkpoint
        bundles.append(dict(item, bundle=str(path), run_config=str(run_path), contract=contract))
        model_manifest[f'{family}:{seed}'] = dict(manifest=receipt, run_config_sha256=sha256(run_path),
            manifest_sha256=sha256(path / 'manifest.json'))
    if purpose == 'validation':
        for family in {b['family'] for b in bundles}:
            if {b['training_seed'] for b in bundles if b['family'] == family} != set(protocol.TRAINING_SEEDS):
                raise ValueError('validation requires all three training seeds per family')
    for diagnostic in diagnostics:
        family = 'J_' + diagnostic[-1]
        selected = [b for b in bundles if b['family'] == family]
        if not selected:
            raise ValueError('diagnostic missing its original joint bundle')
        for item in selected:
            checkpoints[(diagnostic, item['training_seed'])] = checkpoints[(family, item['training_seed'])]
    if test_inputs and plan['panel'] == 'diagnostics':
        checkpoints = {k: v for k, v in checkpoints.items() if k[0] in protocol.DIAGNOSTICS}
    legacy_manifest = None
    legacy_cache = plan.get('legacy_cache')
    if set(fixed) & {'G_old', 'M_old', 'C_old'}:
        if not legacy_cache:
            raise ValueError('requested legacy modes require explicit numeric cache')
        legacy_cache = str(Path(legacy_cache).resolve())
        _, legacy_manifest = load_legacy(legacy_cache)
        for entry in legacy_manifest['models'].values():
            for key in _BINDING_FIELDS:
                if entry['config'][key] != physical[key] and not (purpose == 'development' and key == 'road_length'):
                    raise ValueError(f'legacy physical mismatch: {key}')
    elif legacy_cache:
        raise ValueError('unused legacy cache')
    for method in fixed:
        checkpoints[(method, None)] = (json_hash(legacy_manifest['models'][method]) if method.endswith('_old')
                                      else json_hash(dict(method=method, config=physical, rule_detection=30, rule_cooldown=1)))
    if not checkpoints:
        raise ValueError('empty requested matrix')
    plan.update(seed_manifest=[r._asdict() for r in expected], physical=physical, legacy_cache=legacy_cache)
    provenance = dict(plan=plan, models=model_manifest, legacy=legacy_manifest, deviations=deviations,
        sources={str(p.relative_to(ROOT)): sha256(p) for p in [
            *sorted((ROOT / 'src/snfs_traffic').rglob('*.py')),
            ROOT / 'research/evaluate_pv_graph_ppo.py',
            *(ROOT / 'sep_research' / n for n in ('evaluate.py', 'export_legacy.py', 'env_core.py',
                                                'observations.py', 'inference.py', 'checkpoints.py', 'protocol.py', 'analysis.py',
                                                'EXPERIMENT_PROTOCOL.md', 'TECHNICAL_SPEC.md'))]},
        runtime=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__),
        science_status='development_smoke_not_scientific_results' if purpose == 'development' else 'validation_candidate_not_selected',
        heldout_test_ready=False)
    if test_inputs:
        proof = _test_matrix(*test_inputs)
        ordered = {(r['method'], r['training_seed']): r['joint_checkpoint_id'] for r in proof['matrix']}
        if checkpoints != ordered:
            raise ValueError('heldout loaded checkpoint matrix mismatch')
        checkpoints = ordered
        provenance.update(heldout_gate=proof,
                          science_status='heldout_execution_authorized_not_completion')
    prepared = dict(plan=plan, manifest=expected, checkpoints=checkpoints, bundles=bundles,
                provenance=provenance, binding=json_hash(provenance),
                config_hash=json_hash(physical), protocol_hash=sha256(ROOT / 'sep_research/EXPERIMENT_PROTOCOL.md'))

    if test_inputs:
        _check_test_prepared(prepared, *test_inputs)
    return prepared


def _worker_init(prepared):
    global _MODELS, _LEGACY
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _MODELS = {}
    for item in prepared['bundles']:
        models, manifest = load_pair(item['bundle'], item['contract'])
        if manifest != prepared['provenance']['models'][f"{item['family']}:{item['training_seed']}"]['manifest']:
            raise ValueError('bundle changed since preflight')
        _MODELS[(item['family'], item['training_seed'])] = models
    _LEGACY = {}
    if prepared['plan']['legacy_cache']:
        _LEGACY, manifest = load_legacy(prepared['plan']['legacy_cache'])
        if manifest != prepared['provenance']['legacy']:
            raise ValueError('legacy cache changed since preflight')


def arrays(state):
    """Explicit TrafficState mapping for the protocol, never object serialization."""
    return {name: getattr(state, name) for name in protocol.PHYSICAL_FIELDS + protocol.ROLE_FIELDS}


def _fingerprint(state, pv, av=()):
    return protocol.state_fingerprints(arrays(state), priority_vehicle_id=pv, av_vehicle_ids=av)


def validate_tags(warm, pre, activated, pv, av):
    """Check source tags and all physical arrays before attesting row provenance."""
    chosen = pre.vehicle_id == pv
    if chosen.sum() != 1 or not pre.alive[chosen].all() or not pre.alive.all():
        raise ValueError('PV must be unique and alive')
    if (np.any(warm.veh_type != 0) or np.any(warm.behavior_id != HDV_BEHAVIOR_ID) or np.any(warm.controlled) or np.any(pre.controlled)
            or np.any(activated.controlled) or np.any(pre.veh_type[~chosen] != 0)
            or np.any(pre.veh_type[chosen] != PRIORITY_VEH_TYPE)
            or np.any(pre.behavior_id[chosen] != PRIORITY_BEHAVIOR_ID)
            or not np.array_equal(high_speed_vehicle_mask(pre), chosen)):
        raise ValueError('invalid warm/PV vmax role tags')
    for name in protocol.PHYSICAL_FIELDS:
        if not (np.array_equal(getattr(warm, name), getattr(pre, name))
                and np.array_equal(getattr(pre, name), getattr(activated, name))):
            raise ValueError('assignment or AV tagging changed physical arrays')
    if not np.array_equal(warm.behavior_id[~chosen], pre.behavior_id[~chosen]):
        raise ValueError('pre-AV background behavior changed')
    selected = np.isin(pre.vehicle_id, av)
    if len(set(av)) != len(av) or selected.sum() != len(av) or np.any(selected & chosen):
        raise ValueError('invalid selected AV IDs')
    expected_type, expected_behavior = pre.veh_type.copy(), pre.behavior_id.copy()
    expected_type[selected], expected_behavior[selected] = AV_VEH_TYPE, PRIORITY_YIELD_BEHAVIOR_ID
    if not (np.array_equal(activated.veh_type, expected_type)
            and np.array_equal(activated.behavior_id, expected_behavior)
            and np.array_equal(high_speed_vehicle_mask(activated), chosen)):
        raise ValueError('activated tags differ from exact allowed AV subset')


def prepare_group(physical, seed_rows):
    """One original all-HDV warmup, one PV assignment, nested AV subsets."""
    row = seed_rows[0]
    warm, backend = graph_eval._warm_snapshot(physical, graph_eval._make_params(physical),
        density=row.density_token / 100, scenario_seed=row.scenario_seed, warmup_rng_seed=row.warmup_rng_seed)
    assigned = assign_priority_vehicles(warm, PriorityVehicleConfig(count=1, placement='random', seed=row.assignment_seed),
                                        road_length=physical['road_length'])
    pre, pv = assigned.state, assigned.vehicle_ids[0]
    states, previous = {}, set()
    for row in sorted(seed_rows, key=lambda r: r.fraction_token):
        activated, av = select_yielding_vehicle_ids(pre, fraction=row.fraction_token / 100, seed=row.av_selection_seed)
        validate_tags(warm, pre, activated, pv, av)
        if not previous <= set(av):
            raise ValueError('AV subsets not nested')
        previous = set(av)
        states[row.fraction_token] = (activated, av)
    validate_tags(warm, pre, pre, pv, ())
    return warm, pre, pv, states, backend


def composition(method):
    """Return core mode and PV/AV execution without changing weight ownership."""
    if method in protocol.FIXED_CONTROLS:
        return 'A', 'native', method
    if method in protocol.FAMILIES:
        return method, 'native' if method == 'A' else 'learned', 'native' if method.startswith('P_') else 'learned'
    if method in protocol.DIAGNOSTICS:
        family = 'J_' + method[-1]
        return family, {'1': 'learned', '2': 'native', '3': 'stay', '4': 'left'}[method[1]], 'native' if method[1] == '1' else 'learned'
    raise ValueError('unknown method')


def measure(prepared, seed_row, method, training_seed, initial, pv, av):
    physical = prepared['plan']['physical']
    mode, pv_mode, av_mode = composition(method)
    core = SepResearchCore(dict(physical, episode_steps=physical['measure_steps'], warmup_cache=False,
        controller_mode=mode, pv_encoder='graph_mlp' if mode.endswith('_G') else 'local_mlp'))
    try:
        metadata = {f: getattr(seed_row, f) for f in protocol.SEED_FIELDS}
        obs, infos = core.reset_from_snapshot(initial, priority_vehicle_id=pv, **metadata,
            density=seed_row.density_token / 100, av_fraction=seed_row.fraction_token / 100,
            warmup_steps=physical['warmup_steps'])
        if any(infos['pv'][k] != v for k, v in metadata.items()) or core.controller_mode != mode:
            raise ValueError('installed mode/seed binding mismatch')
        if any(not np.array_equal(getattr(core.state, f.name), getattr(initial, f.name)) for f in fields(initial)):
            raise ValueError('snapshot install changed state')
        if (core.params.vmax_default, core.params.vmax_controlled) != (5, 6):
            raise ValueError('speed contract mismatch')
        models = _MODELS.get((mode, training_seed), {})
        if ((pv_mode == 'learned' and models.get('pv_policy') is None)
                or (av_mode == 'learned' and models.get('av_policy') is None)):
            raise ValueError('missing bound role model')
        controller = PriorityYieldController(PriorityYieldConfig(detection_distance=30, cooldown_steps=1),
            eligible_vehicle_ids=av) if method == 'B2' else None
        metrics = PriorityMetricsAccumulator(core.params, priority_vehicle_ids=(pv,))
        events = {role: defaultdict(int) for role in core.possible_agents}
        reasons = {role: defaultdict(int) for role in core.possible_agents}
        paths, active_counts = set(), []
        pv_lanes, pv_speeds = [0] * 4, [0] * 7
        empty = stay_only = reference_steps = reversals = pv_applied = 0
        previous_delta = 0
        for _ in range(physical['measure_steps']):
            graph = core._article._current_graph
            active_counts.append(graph.active_count)
            empty += int(graph.active_count == 0)
            stay_only += int(np.sum(obs['pv']['action_mask']) == 1)
            commands, decision = {}, None
            if pv_mode != 'native':
                index = (int(predict(models['pv_policy'], obs['pv']).action) if pv_mode == 'learned'
                         else 0 if pv_mode == 'left' and obs['pv']['action_mask'][0] else 1)
                commands['pv'] = {pv: index - 1}
            if controller:
                decision = controller.decide(core.state, core.params, core.simulator.topology)
                commands['av_coordinator'] = decision.overrides
            elif av_mode not in ('native', 'B1'):
                raw = obs['av_coordinator']
                if av_mode == 'learned':
                    action = predict(models['av_policy'], raw).action.numpy()
                elif av_mode in _LEGACY:
                    action = legacy_prediction(_LEGACY[av_mode], raw)[2]
                else:
                    action = np.where(raw['action_mask'][:, 0], 0, 1) if av_mode == 'L' else np.ones(164, dtype=np.int64)
                commands['av_coordinator'] = {int(vid): int(a) - 1 for vid, a in zip(graph.vehicle_id[:graph.active_count], action)}
            obs, _, _, truncated, infos = core.step_commands(commands, apply_cooldowns=controller is None)
            info = infos['pv']
            state, result = core.state, core.simulator.last_step_info.action_result
            if controller and decision.overrides:
                controller.observe(result)
            metrics.observe(state, decision=decision,
                            result=result if controller and decision.overrides else None)
            for role, counts in info['role_events'].items():
                for key, value in counts.items():
                    events[role][key] += value
            for vid, delta, applied, reason in zip(result.vehicle_id, result.requested_lane_delta, result.applied, result.rejection_reason):
                if delta and not applied:
                    reasons['pv' if int(vid) == pv else 'av_coordinator'][reason] += 1
                if int(vid) == pv and delta and applied:
                    reversals += int(previous_delta != 0 and previous_delta != int(delta))
                    previous_delta = int(delta)
                    pv_applied += 1
            paths.add(info['backend_name'])
            reference_steps += int(info['used_reference_action_path'])
            idx = np.flatnonzero(state.vehicle_id == pv)[0]
            pv_lanes[int(state.lane[idx])] += 1
            pv_speeds[int(state.vel[idx])] += 1
        if core.simulator.step_count != physical['measure_steps'] or not truncated['__all__']:
            raise ValueError('physical trajectory horizon mismatch')
        for event in events.values():
            if (event['raw_nonstay'] != event['cooldown'] + event['issued']
                    or event['issued'] != event['applied'] + event['blocked'] + event['conflict']
                    or event['rejected'] != event['blocked'] + event['conflict'] + event['cooldown']):
                raise ValueError('role ledger inconsistency')
        return dict(metrics.summary().to_dict(), role_events={k: dict(v) for k, v in events.items()},
            total_events={k: sum(e[k] for e in events.values()) for k in events['pv']},
            rejection_reasons={k: dict(v) for k, v in reasons.items()}, backend_paths=sorted(paths),
            reference_action_steps=reference_steps, active_av_count_mean=float(np.mean(active_counts)),
            active_av_count_min=min(active_counts), active_av_count_max=max(active_counts),
            av_opportunities=sum(active_counts), pv_opportunities=physical['measure_steps'],
            pv_lane_steps=pv_lanes, pv_speed_steps=pv_speeds, empty_graph_steps=empty,
            pv_stay_only_mask_steps=stay_only, pv_applied_direction_reversals=reversals,
            pv_applied_moves=pv_applied, team_return=info['team_return'],
            physical_steps=core.simulator.step_count, logical_observation_steps=2 * core.simulator.step_count,
            final_state_hash=_fingerprint(core.state, pv, av)['activated_state_hash'])
    finally:
        core.close()


def run_group(prepared, seed_rows):
    warm, pre, pv, states, backend = prepare_group(prepared['plan']['physical'], seed_rows)
    warm_hash = _fingerprint(warm, pv)['activated_state_hash']
    pre_hash = _fingerprint(pre, pv)['activated_state_hash']
    rows, b1 = [], None
    for seed_row in seed_rows:
        for (method, training_seed), checkpoint in prepared['checkpoints'].items():
            initial, av = (pre, ()) if method == 'B1' else states[seed_row.fraction_token]
            hashes = _fingerprint(initial, pv, av)
            if method == 'B1' and b1 is not None:
                result = b1
            else:
                result = measure(prepared, seed_row, method, training_seed, initial, pv, av)
                if method == 'B1':
                    b1 = result
            rows.append(dict(result, **seed_row._asdict(), **hashes,
                method=method, training_seed=training_seed, joint_checkpoint_id=checkpoint,
                warmed_state_hash=warm_hash, pre_av_tag_state_hash=pre_hash,
                priority_vehicle_id=pv, protocol_hash=prepared['protocol_hash'],
                config_hash=prepared['config_hash'], commonconfig_hash=prepared['config_hash'],
                measure_steps=prepared['plan']['physical']['measure_steps'], status='ok',
                role_tags_validated=True, reuse_key=protocol.fixed_reuse_key(method, seed_row) if method in protocol.FIXED_CONTROLS else None,
                av_vehicle_ids=list(av), warm_backend=backend, purpose=prepared['plan']['purpose']))
    if _fingerprint(warm, pv)['activated_state_hash'] != warm_hash or _fingerprint(pre, pv)['activated_state_hash'] != pre_hash:
        raise ValueError('measurement mutated shared warm/assignment state')
    protocol.validate_rows(rows, manifest=seed_rows, checkpoints=prepared['checkpoints'])
    return rows


def _atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', prefix='.' + path.name + '.partial_', dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temp, path)  # Atomic and exclusive; never overwrites a complete chunk.
        finally:
            temp.unlink(missing_ok=True)


def _read_json(path):
    """Reject ambiguous JSON and non-files without modifying outputs."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'missing or unsafe output: {path.name}')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'duplicate JSON key: {key}')
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError(f'nonfinite JSON: {value}')
    return json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=nonfinite)


def _groups(prepared):
    groups = defaultdict(list)
    for row in prepared['manifest']:
        groups[(row.canonical_density_id, row.run_index)].append(row)
    return groups


def _verify_provenance(out, prepared):
    expected = dict(prepared['provenance'], binding=prepared['binding'])
    actual = _read_json(out / 'provenance.json')
    if (json_hash(prepared['provenance']) != prepared['binding']
            or json_hash(actual) != json_hash(expected)):
        raise ValueError('resume plan/model/seed/source/runtime mismatch')


def _validate_output_rows(rows, prepared, manifest):
    protocol.validate_rows(rows, manifest=manifest, checkpoints=prepared['checkpoints'])
    physical = prepared['plan']['physical']
    expected = dict(config_hash=prepared['config_hash'], commonconfig_hash=prepared['config_hash'],
        protocol_hash=prepared['protocol_hash'], purpose=prepared['plan']['purpose'],
        measure_steps=physical['measure_steps'], physical_steps=physical['measure_steps'],
        logical_observation_steps=2 * physical['measure_steps'],
        pv_opportunities=physical['measure_steps'], role_tags_validated=True)
    reused, subsets = {}, {}
    for row in rows:
        for field, value in expected.items():
            if field not in row or type(row[field]) is not type(value) or row[field] != value:
                raise ValueError(f'row expected contract mismatch: {field}')
        av = row.get('av_vehicle_ids')
        if (not isinstance(av, list) or any(type(v) is not int or v < 0 for v in av)
                or len(set(av)) != len(av) or row['priority_vehicle_id'] in av
                or protocol._array_hash((('av_vehicle_ids', np.asarray(av, dtype=np.int64)),)) != row['av_subset_hash']):
            raise ValueError('row AV subset mismatch')
        if row['method'] == 'B1':
            if av or row['activated_state_hash'] != row['pre_av_tag_state_hash']:
                raise ValueError('B1 native role mismatch')
        else:
            key = (row['canonical_density_id'], row['run_index'], row['fraction_token'])
            if subsets.setdefault(key, av) != av:
                raise ValueError('row role subset mismatch')
        if row['reuse_key'] is not None:
            # Only B1's logical fraction changes for a reused physical trajectory.
            signature = {k: v for k, v in row.items() if k != 'fraction_token'}
            if reused.setdefault(row['reuse_key'], signature) != signature:
                raise ValueError('reused fixed trajectory field mismatch')
    previous = {}
    for (density, run, fraction), av in sorted(subsets.items()):
        if not previous.get((density, run), set()) <= set(av):
            raise ValueError('row AV subsets not nested')
        previous[(density, run)] = set(av)


def _read_chunk(path, prepared, seed_rows):
    chunk = _read_json(path)
    # Current case identity is the filename, not a case_key field. Validate its
    # exact manifest group; unknown chunk metadata is not part of this format.
    if (set(chunk) != {'status', 'binding', 'rows_hash', 'rows'}
            or chunk['status'] != 'complete' or chunk['binding'] != prepared['binding']
            or chunk['rows_hash'] != json_hash(chunk['rows'])):
        raise ValueError('corrupt or mixed complete chunk')
    _validate_output_rows(chunk['rows'], prepared, seed_rows)
    return chunk['rows']


def _counts(prepared, rows, groups):
    trajectories = len(rows) - (len(prepared['manifest']) - len(groups) if ('B1', None) in prepared['checkpoints'] else 0)
    return dict(logical_rows=len(rows), physical_trajectories=trajectories,
        measurement_steps=trajectories * prepared['plan']['physical']['measure_steps'],
        warm_trajectories=len(groups), warm_steps=len(groups) * prepared['plan']['physical']['warmup_steps'],
        science_status=prepared['provenance']['science_status'], heldout_test_ready=False)


def _writer_json_partial(path, authoritative_names):
    # tempfile's current eight-character suffix; never evidence of completion.
    match = re.fullmatch(r'\.(.+\.json)\.partial_[a-z0-9_]{8}', path.name)
    return bool(match and match[1] in authoritative_names and path.is_file())


def _verify_reader_layout(out, prepared, groups):
    required = {'provenance.json', 'chunks', 'raw.json', 'raw.csv', 'counts.json'}
    if not required <= {p.name for p in out.iterdir()}:
        raise ValueError('missing or unexpected output files')
    failures, partials = set(), []
    for path in out.iterdir():
        if path.is_symlink():
            raise ValueError('unsafe output file')
        if path.name in required | {'.lock'}:
            if not (path.is_dir() if path.name == 'chunks' else path.is_file()):
                raise ValueError('unsafe output file')
        elif re.fullmatch(r'failed_(0|[1-9][0-9]*)_(0|[1-9][0-9]*)_(0|[1-9][0-9]*)\.json', path.name):
            _, density, run, _ = path.stem.split('_')
            key = (int(density), int(run))
            failure = _read_json(path)
            if (key not in groups or not isinstance(failure, dict)
                    or not isinstance(failure.get('error'), str)
                    or json_hash(failure) != json_hash(dict(status='failed', binding=prepared['binding'],
                        group=list(key), error=failure['error'], seed_rows=[r._asdict() for r in groups[key]]))):
                raise ValueError('corrupt or foreign failure artifact')
            failures.add(path.name)
        elif re.fullmatch(r'\.csv\.partial_[a-z0-9_]{8}', path.name):
            if not path.is_dir() or any(p.name != 'raw.csv' or p.is_symlink() or not p.is_file()
                                         for p in path.iterdir()):
                raise ValueError('unsafe CSV partial')
        else:
            partials.append(path)
    # Every allowed counterpart is independently verified, including all chunks below.
    authoritative = (required - {'chunks', 'raw.csv'}) | failures
    if any(not _writer_json_partial(path, authoritative) for path in partials):
        raise ValueError('unexpected output files')
    chunks = out / 'chunks'
    expected = {f'{d}_{r}.json' for d, r in groups}
    if not expected <= {p.name for p in chunks.iterdir()}:
        raise ValueError('missing or unexpected chunks')
    for path in chunks.iterdir():
        if path.is_symlink() or (path.name not in expected and not _writer_json_partial(path, expected)):
            raise ValueError('unsafe or unexpected chunk')


def read_evaluation(out_dir, *, expected_prepared, expected_binding=None):
    """Return verified rows from one complete invocation, strictly without writes.

    expected_prepared MUST be the trusted caller's independently obtained
    prepare_plan(expected_plan) result, retained in memory (or transported by a
    trusted caller). Never derive it from this output's provenance. Preparation
    validates numeric bundles. For development/validation, the reader checks
    output JSON/CSV against that snapshot without re-reading inputs or restoring
    models. It never invokes evaluation, pickle, Ray or simulation.

    Test revalidates the designated profile and full selection, then checks the
    exact panel and gated proof without restoring heldout models. Selection may
    CPU-load validation candidates for offline verification.
    Counts retain the evaluator's original warm/measurement accounting.
    """
    prepared = expected_prepared
    if prepared['plan'].get('purpose') == 'test':
        if 'heldout_gate' not in prepared.get('provenance', {}):
            raise ValueError('test requires selection gated preparation')
        plan = prepared['plan']
        _check_test_prepared(prepared, *_test_inputs(plan['selection_receipt'], plan['panel'], expected_plan=plan))
    if prepared['plan'].get('purpose') not in ('development', 'validation', 'test'):
        raise ValueError('purpose must be development, validation or test')
    if (prepared['plan'] != prepared['provenance']['plan']
            or prepared['config_hash'] != json_hash(prepared['plan']['physical'])
            or prepared['protocol_hash'] != prepared['provenance']['sources']['sep_research/EXPERIMENT_PROTOCOL.md']
            or [r._asdict() for r in prepared['manifest']] != prepared['plan']['seed_manifest']
            or (expected_binding is not None and expected_binding != prepared['binding'])):
        raise ValueError('expected preparation mismatch')
    out = Path(out_dir)
    if out.is_symlink() or not out.is_dir():
        raise ValueError('missing or unsafe output directory')
    groups = _groups(prepared)
    _verify_reader_layout(out, prepared, groups)
    _verify_provenance(out, prepared)
    chunks = out / 'chunks'
    rows = [row for key in sorted(groups)
            for row in _read_chunk(chunks / f'{key[0]}_{key[1]}.json', prepared, groups[key])]
    _validate_output_rows(rows, prepared, prepared['manifest'])
    for name, value in (('raw.json', rows), ('counts.json', _counts(prepared, rows, groups))):
        if json_hash(_read_json(out / name)) != json_hash(value):
            raise ValueError(f'existing output mismatch: {name}')
    # Original writer semantics: nulls, nested JSON, floats, field union and CRLF.
    fields = list(dict.fromkeys(k for row in rows for k in sorted(row)))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: graph_eval._csv_cell(row.get(k)) for k in fields})
    if (out / 'raw.csv').read_bytes() != stream.getvalue().encode('utf-8'):
        raise ValueError('existing CSV mismatch')
    return rows


def evaluate(plan, out_dir, *, workers=1, resume=False):
    _positive(workers, 'workers')
    prepared = prepare_plan(plan)
    out = Path(out_dir).resolve()
    if out.exists() and not resume:
        raise FileExistsError(out)
    if not out.exists():
        if resume:
            raise ValueError('resume requires existing provenance')
        out.mkdir(parents=True)
        _atomic_json(out / 'provenance.json', dict(prepared['provenance'], binding=prepared['binding']))
        (out / 'chunks').mkdir()
    _verify_provenance(out, prepared)
    lock = (out / '.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError('another evaluator owns this output') from None
    try:
        groups = _groups(prepared)
        expected_names = {f'{d}_{r}.json' for d, r in groups}
        if any(p.name not in expected_names and '.partial_' not in p.name for p in (out / 'chunks').iterdir()):
            raise ValueError('unexpected chunk')
        completed, pending = {}, {}
        # Validate ALL existing chunks before scheduling any new simulation.
        for key, seed_rows in groups.items():
            path = out / 'chunks' / f'{key[0]}_{key[1]}.json'
            if not path.exists():
                pending[key] = seed_rows
                continue
            completed[key] = _read_chunk(path, prepared, seed_rows)
        if pending and any((out / name).exists() for name in ('raw.json', 'raw.csv', 'counts.json')):
            raise ValueError('complete outputs with missing complete chunks')
        def publish(key, rows):
            _validate_output_rows(rows, prepared, groups[key])
            _atomic_json(out / 'chunks' / f'{key[0]}_{key[1]}.json',
                         dict(status='complete', binding=prepared['binding'], rows_hash=json_hash(rows), rows=rows))
            completed[key] = rows
        def failed(key, exc):
            # Unique failure artifact retains reason without accepting partial trajectories.
            file = out / f'failed_{key[0]}_{key[1]}_{len(list(out.glob("failed_*.json")))}.json'
            _atomic_json(file, dict(status='failed', binding=prepared['binding'], group=list(key),
                                   error=repr(exc), seed_rows=[r._asdict() for r in groups[key]]))
        if pending and workers == 1:
            _worker_init(prepared)
            for key, seed_rows in pending.items():
                try:
                    publish(key, run_group(prepared, seed_rows))
                except Exception as exc:
                    failed(key, exc)
                    raise
        elif pending:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                    initializer=_worker_init, initargs=(prepared,)) as pool:
                futures = {pool.submit(run_group, prepared, seed_rows): key for key, seed_rows in pending.items()}
                for future in as_completed(futures):
                    key = futures[future]
                    try:
                        publish(key, future.result())
                    except Exception as exc:
                        failed(key, exc)
                        for task in futures:
                            task.cancel()
                        raise
        rows = [row for key in sorted(completed) for row in completed[key]]
        _validate_output_rows(rows, prepared, prepared['manifest'])
        counts = _counts(prepared, rows, groups)
        for name, value in (('raw.json', rows), ('counts.json', counts)):
            path = out / name
            if path.exists():
                if json.loads(path.read_text()) != value:
                    raise ValueError('existing output mismatch')
            else:
                _atomic_json(path, value)
        # Use the original JSON-in-CSV cell convention, publishing exclusively.
        with tempfile.TemporaryDirectory(prefix='.csv.partial_', dir=out) as directory:
            temp = Path(directory) / 'raw.csv'
            graph_eval._write_csv(temp, [{k: row[k] for k in sorted(row)} for row in rows])
            target = out / 'raw.csv'
            if target.exists():
                if target.read_bytes() != temp.read_bytes():
                    raise ValueError('existing CSV mismatch')
            else:
                os.link(temp, target)
        return rows
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    evaluate(json.loads(args.plan.read_text()), args.out_dir, workers=args.workers, resume=args.resume)


if __name__ == '__main__':
    main()
