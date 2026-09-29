"""Bounded, forward-only acceptance of the immutable evaluation snapshot.

Run directly with python -I; no pytest collection or current test imports.
Fixtures/assertions adapted from the reviewed test_observations, test_inference,
test_snfs_full_step, test_env_core and test_env sources; these are NEW tests.
"""
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import unittest
import xml.etree.ElementTree as ET
from dataclasses import fields

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'build/verification'
FROZEN = ROOT / 'code'
PROVENANCE = ROOT / 'provenance'
sys.dont_write_bytecode = True
assert sys.flags.isolated, 'Run with python -I'
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
sys.path[:] = [str(FROZEN / 'src'), str(FROZEN)] + [
    p for p in sys.path if p and not Path(p).resolve().is_relative_to(ROOT)]
MANIFEST = json.loads((PROVENANCE / 'science_sources.json').read_text())
for relative, digest in MANIFEST['sources'].items():
    assert hashlib.sha256((FROZEN / relative).read_bytes()).hexdigest() == digest, relative
OUT.mkdir(parents=True, exist_ok=True)

import numpy as np
import torch
from sep_research.observations import build_pv_ego_observation
from sep_research.env_core import SepResearchCore
from sep_research.models import PvManeuverTorchModel, PvGraphManeuverTorchModel
from sep_research.inference import build_av_model, build_pv_model, predict
from snfs_traffic.core import (SimulationParams, empty_state, build_occupancy,
    step_lane_change_reference, step_longitudinal_reference, step_reference,
    validate_runtime_invariants)
from snfs_traffic.core import longitudinal_numba as ln
from snfs_traffic.observations.pv_graph import PvGraphConfig
from snfs_traffic.rl.graph_ppo import PvGraphTorchModel
from snfs_traffic.scenarios import PRIORITY_VEH_TYPE, AV_VEH_TYPE
from snfs_traffic.simulator import TrafficSimulator
from snfs_traffic.topology import RingTopology
from snfs_traffic.rules import PriorityYieldController, PriorityYieldConfig

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
TRANSITIONS = 0
EVIDENCE = {}


def census():
    result = {}
    for name, module in sorted(sys.modules.items()):
        if not (name == 'sep_research' or name.startswith('sep_research.')
                or name == 'snfs_traffic' or name.startswith('snfs_traffic.')):
            continue
        path = Path(module.__file__).resolve()
        relative = str(path.relative_to(FROZEN))  # Reject editable/current leakage.
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == MANIFEST['sources'][relative], (name, relative)
        result[name] = dict(file=str(path), sha256=digest)
    return result


census()  # Gate before scientific test execution, in this same process.


def state(lanes, positions, speeds, types=None):
    s = empty_state(len(lanes))
    s.lane[:], s.pos[:], s.vel[:] = lanes, positions, speeds
    if types is not None:
        s.veh_type[:] = types
    return s


def equal(a, b):
    for f in fields(a):
        np.testing.assert_array_equal(getattr(a, f.name), getattr(b, f.name), err_msg=f.name)


def core(s=None, **kw):
    env = SepResearchCore(dict(road_length=100, density=.15, av_fraction=1,
        warmup_steps=0, warmup_cache=False, episode_steps=10, backend='reference', **kw))
    if s is None:
        s = state([1, 1, 2], [10, 18, 50], [3, 2, 1],
                  [PRIORITY_VEH_TYPE, AV_VEH_TYPE, 0])
    obs, _ = env.reset_from_snapshot(s, priority_vehicle_id=0, scenario_seed=11,
        warmup_rng_seed=12, assignment_seed=13, av_selection_seed=14,
        measurement_rng_seed=15, density=.15, av_fraction=1, warmup_steps=0)
    return env, obs


def transition(call, *args, **kw):
    global TRANSITIONS
    TRANSITIONS += 1  # Counts attempted calls, conservatively includes phase calls.
    assert TRANSITIONS <= 40
    return call(*args, **kw)


def tensors(obs):
    return {k: tensors(v) if isinstance(v, dict) else torch.as_tensor(v).unsqueeze(0)
            for k, v in obs.items()}


class MinimalAcceptance(unittest.TestCase):
    def test_t1_pv48_manual_wrap_missing(self):
        s = state([0, 0, 1, 1, 2], [199, 0, 59, 138, 190], [4, 2, 3, 1, 5],
                  [PRIORITY_VEH_TYPE, 0, 0, 0, AV_VEH_TYPE])
        before = s.copy()
        obs = build_pv_ego_observation(s, 0, SimulationParams(num_lanes=4, road_length=200),
                                       PvGraphConfig(), {0: 1, 4: 1})
        expected = np.zeros(48, dtype=np.float32)
        expected[3], expected[6], expected[18], expected[29] = 4/6, 1, 1, 1
        expected[19:24] = [1, 1/60, -2/6, 1, 0]
        expected[30:35] = [1, 1, -1/6, 1, 0]
        expected[40:44] = [1, 1/60, 0, -2/6]
        np.testing.assert_array_equal(obs['ego_features'], expected)
        self.assertEqual(obs['ego_features'].dtype, np.float32)
        np.testing.assert_array_equal(obs['action_mask'], [0, 1, 0])
        equal(s, before)

    def parity(self, kind, count):
        torch.manual_seed(731)
        for empty in (False, True):
            s = state([1, 1, 2], [10, 18, 50], [3, 2, 1],
                      [PRIORITY_VEH_TYPE, 0 if empty else AV_VEH_TYPE, 0])
            env, observations = core(s, pv_encoder='graph_mlp' if kind == 'graph_mlp' else 'local_mlp')
            self.addCleanup(env.close)
            graph = observations['av_coordinator']
            self.assertEqual(int(graph['node_mask'].sum()), 0 if empty else 1)
            padding = graph['node_mask'] == 0
            self.assertFalse(graph['node_features'][padding].any())
            self.assertFalse(graph['neighbor_mask'][padding].any())
            np.testing.assert_array_equal(graph['action_mask'][padding],
                np.tile([0, 1, 0], (int(padding.sum()), 1)))
            if kind == 'av':
                original = PvGraphTorchModel(env.observation_spaces['av_coordinator'],
                    env.action_spaces['av_coordinator'], 492,
                    {'custom_model_config': {'hidden_dim': 128, 'message_layers': 3}}, 'synthetic')
                obs = graph
            else:
                cls = PvGraphManeuverTorchModel if kind == 'graph_mlp' else PvManeuverTorchModel
                original = cls(env.observation_spaces['pv'], env.action_spaces['pv'], 3, {}, 'synthetic')
                obs = observations['pv']
            original.cpu().eval()
            weights = {k: v.detach().numpy().copy() for k, v in original.state_dict().items()}
            rng = torch.random.get_rng_state().clone()
            model = build_av_model(weights, {}) if kind == 'av' else build_pv_model(weights, kind)
            self.assertEqual(sum(p.numel() for p in original.parameters()), count)
            self.assertEqual(sum(p.numel() for p in model.parameters()), count)
            before = copy.deepcopy(obs)
            with torch.no_grad():
                expected, _ = original.forward({'obs': tensors(obs)}, [], None)
                value = original.value_function()[0]
            actual = predict(model, obs)  # NumPy input AND synthetic NumPy state_dict.
            torch.testing.assert_close(actual.logits.flatten(), expected.flatten(), atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(actual.value, value, atol=1e-6, rtol=1e-6)
            self.assertEqual(tuple(actual.logits.shape), (164, 3) if kind == 'av' else (3,))
            self.assertEqual(tuple(actual.value.shape), ())
            expected_actions = expected.reshape(-1, 3).argmax(-1).reshape(actual.action.shape)
            torch.testing.assert_close(actual.action, expected_actions, atol=0, rtol=0)
            self.assertTrue((actual.logits[~torch.as_tensor(obs['action_mask']).bool()] == -1e9).all())
            if kind != 'local_mlp':
                changed = copy.deepcopy(obs)
                changed_graph = changed if kind == 'av' else changed['av_graph']
                changed_graph['node_features'][padding] = .75
                changed_result = predict(model, changed)
                torch.testing.assert_close(actual.value, changed_result.value, atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(actual.logits, changed_result.logits, atol=1e-6, rtol=1e-6)
            for key in obs:
                if isinstance(obs[key], dict):
                    for subkey in obs[key]:
                        np.testing.assert_array_equal(obs[key][subkey], before[key][subkey])
                else:
                    np.testing.assert_array_equal(obs[key], before[key])
            torch.testing.assert_close(torch.random.get_rng_state(), rng, atol=0, rtol=0)

    def test_t1_local23812(self):
        self.parity('local_mlp', 23812)

    def test_t1_graph411012(self):
        self.parity('graph_mlp', 411012)

    def test_t1_av355460(self):
        self.parity('av', 355460)

    def test_t2_phase_occupancy_ids(self):
        s = state([0, 0], [10, 11], [3, 0])
        before = s.copy()
        p = SimulationParams(num_lanes=2, road_length=20, p_lane_change=1,
                             r=0, P2=1, P3=1, P4=0)
        t = RingTopology(2, 20)
        rng = np.random.default_rng(0)
        lc = transition(step_lane_change_reference, s, p, t, rng)
        np.testing.assert_array_equal(lc.pos, s.pos)
        np.testing.assert_array_equal(lc.vel, s.vel)
        self.assertEqual(int(lc.lane[0]), 1)
        manual = transition(step_longitudinal_reference, lc, p, t, rng)
        manual.changed_lane, manual.last_lane_delta = lc.changed_lane.copy(), lc.last_lane_delta.copy()
        other_rng = np.random.default_rng(0)
        full = transition(step_reference, s, p, t, other_rng)
        equal(full, manual)
        equal(s, before)
        self.assertEqual((int(full.lane[0]), int(full.vel[0]), int(full.pos[0])), (1, 4, 14))
        np.testing.assert_array_equal(full.vehicle_id, s.vehicle_id)
        np.testing.assert_array_equal(full.alive, s.alive)
        validate_runtime_invariants(full, p, t)
        self.assertEqual(int((build_occupancy(full, p) >= 0).sum()), 2)
        self.assertEqual(rng.bit_generator.state, other_rng.bit_generator.state)

    def test_t2_sparse_applied_rejected(self):
        for conflict in (False, True):
            s = state([0, 2], [10, 10 if conflict else 50], [0, 0],
                      [PRIORITY_VEH_TYPE, AV_VEH_TYPE])
            env, _ = core(s)
            self.addCleanup(env.close)
            commands = {'pv': {0: 1}, 'av_coordinator': {1: -1}} if conflict else {'pv': {0: 1}}
            result = transition(env.step_commands, commands)
            events = result[4]['pv']['role_events']
            self.assertEqual(events['pv']['applied'], 0 if conflict else 1)
            self.assertEqual(events['pv']['rejected'], 1 if conflict else 0)
            if conflict:
                self.assertEqual(env.simulator.last_step_info.action_result.rejection_reason,
                                 ('override_conflict', 'override_conflict'))
                np.testing.assert_array_equal(env.state.lane, [0, 2])
            else:
                self.assertEqual(int(env.state.lane[0]), 1)
            self.assertEqual(env.simulator.step_count, 1)
            self.assertEqual(result[1]['pv'], result[4]['pv']['pv_speed']/6 - (.002 if conflict else 0))
            np.testing.assert_array_equal(env.state.vehicle_id, s.vehicle_id)
            validate_runtime_invariants(env.state, env.params, env.simulator.topology)

    def test_t2_native_vs_explicit_stay(self):
        s = state([1, 1], [10, 12], [3, 0], [PRIORITY_VEH_TYPE, 0])
        native, _ = core(s, p_lane_change=1)
        stay, _ = core(s, p_lane_change=1)
        self.addCleanup(native.close)
        self.addCleanup(stay.close)
        oracle = TrafficSimulator(params=native.params, backend='reference')
        oracle.reset(state=s, rng_seed=15)
        output = transition(native.step_commands, {})
        transition(stay.step_commands, {'pv': {0: 0}})
        expected = transition(oracle.step)
        equal(native.state, expected)
        self.assertEqual(native.simulator._rng.bit_generator.state, oracle._rng.bit_generator.state)
        self.assertNotEqual(int(native.state.lane[0]), 1)
        self.assertEqual(int(stay.state.lane[0]), 1)
        self.assertEqual(output[4]['pv']['total_events']['raw_stay'], 0)

    def test_t2_b2_c1(self):
        s = state([1, 1], [10, 18], [3, 2], [PRIORITY_VEH_TYPE, AV_VEH_TYPE])
        env, _ = core(s, cooldown_steps=9)
        self.addCleanup(env.close)
        ctl = PriorityYieldController(PriorityYieldConfig(cooldown_steps=1), eligible_vehicle_ids=[1])
        first = ctl.decide(s, env.params, env.simulator.topology)
        self.assertEqual(dict(first.overrides), {1: 1})  # Odd ID: equal-clearance right tie.
        env._article._cooldown_by_vehicle_id = {1: 9}
        out = transition(env.step_commands, {'av_coordinator': first.overrides}, apply_cooldowns=False)
        self.assertEqual(out[4]['pv']['role_events']['av_coordinator']['applied'], 1)
        ctl.observe(env.simulator.last_step_info.action_result)
        # Fixed trigger state isolates C=1 from motion out of the detection lane.
        second = ctl.decide(s, env.params, env.simulator.topology)
        self.assertEqual(second.cooldown_vehicle_ids, (1,))
        self.assertEqual(dict(second.overrides), {})
        third = ctl.decide(s, env.params, env.simulator.topology)
        self.assertEqual(dict(third.overrides), {1: 1})
        EVIDENCE['b2'] = 'One applied physical transition; C=1 decisions reuse fixed trigger state, not trajectory proof.'

    def test_t2_reference_optimized_compiled(self):
        self.assertTrue(ln.NUMBA_AVAILABLE)
        s = state([0, 0, 0, 1], [38, 2, 14, 20], [3, 0, 2, 1])
        p = SimulationParams(num_lanes=2, road_length=40, p_lane_change=0, q=.65, r=.45,
                             P1=.2, P2=.4, P3=.6, P4=.8)
        sims = [TrafficSimulator(params=p, backend=b) for b in ('reference', 'optimized')]
        for sim in sims:
            sim.reset(state=s, rng_seed=123)
        dispatcher = ln._compute_longitudinal_velocities_unit_length_numba
        calls = []
        # Instrument only the Python dispatch boundary; invoke the original CPUDispatcher.
        def record(*args, **kw):
            result = dispatcher(*args, **kw)
            calls.append({'nopython_signatures': [str(x) for x in dispatcher.nopython_signatures]})
            return result
        ln._compute_longitudinal_velocities_unit_length_numba = record
        try:
            for _ in range(3):
                expected, actual = [transition(sim.step) for sim in sims]
                equal(expected, actual)
                np.testing.assert_array_equal(actual.vehicle_id, s.vehicle_id)
                np.testing.assert_array_equal(actual.alive, s.alive)
                validate_runtime_invariants(actual, p, sims[1].topology)
                self.assertEqual(sims[0]._rng.bit_generator.state, sims[1]._rng.bit_generator.state)
        finally:
            ln._compute_longitudinal_velocities_unit_length_numba = dispatcher
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(x['nopython_signatures'] for x in calls))
        self.assertEqual(sims[1].backend_name, 'optimized')
        EVIDENCE['compiled_longitudinal'] = calls
        EVIDENCE['optimized_scope'] = 'Three native full steps, unit lengths, p_lane_change=0; exact states and RNG.'


if __name__ == '__main__':
    started = time.time()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(MinimalAcceptance)
    ids = [t.id() for t in suite]
    assert len(ids) == 9
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    provenance = census()
    errors = {t.id(): detail for t, detail in result.errors + result.failures}
    skipped = {t.id(): reason for t, reason in result.skipped}
    report = dict(status='PASS' if result.wasSuccessful() and not skipped else 'FAIL',
        elapsed_seconds=time.time()-started, tests=[dict(id=i, status='FAIL' if i in errors else
        'SKIP' if i in skipped else 'PASS', detail=errors.get(i, skipped.get(i))) for i in ids],
        transitions_or_phase_calls=TRANSITIONS, transition_cap=40, evidence=EVIDENCE,
        origins=provenance, versions={name: importlib.metadata.version(name)
            for name in ('numpy', 'torch', 'ray', 'numba', 'gymnasium')},
        python=sys.version, argv=sys.argv, cwd=os.getcwd(), sys_path=sys.path,
        torch_threads=torch.get_num_threads(), cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256((PROVENANCE/'science_sources.json').read_bytes()).hexdigest(),
        criteria_sha256=hashlib.sha256((PROVENANCE/'science_acceptance_criteria.md').read_bytes()).hexdigest())
    (OUT/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    xml = ET.Element('testsuite', name='minimal_acceptance', tests=str(len(ids)),
                     failures=str(len(result.failures)), errors=str(len(result.errors)), skipped=str(len(skipped)))
    for i in ids:
        node = ET.SubElement(xml, 'testcase', name=i)
        if i in errors:
            ET.SubElement(node, 'failure').text = errors[i]
        if i in skipped:
            ET.SubElement(node, 'skipped').text = skipped[i]
    ET.ElementTree(xml).write(OUT/'results.xml', encoding='utf-8', xml_declaration=True)
    print(json.dumps({k: report[k] for k in ('status', 'transitions_or_phase_calls', 'elapsed_seconds')}))
    sys.exit(0 if report['status'] == 'PASS' else 1)
