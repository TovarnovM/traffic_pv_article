"""Explicit published-checkpoint export; numeric CPU readers never import Ray.

CLI: python -m sep_research.export_legacy --plan legacy_plan.json --out-dir DIR
Plan: {"G_old": {"checkpoint": "...", "run_config": "..."}, ... M_old, C_old}.
All three are required. Paths identify exact published checkpoints, never latest.
export_legacy restores trusted local RLlib binaries in isolated subprocesses;
load_legacy verifies cache and original binary/config provenance without unpickling.

The two small non-graph cores mirror snfs_traffic.rl.nongraph_ppo because its
factory imports RLlib. Export checks logits/value/action parity against that
unchanged factory. Graph inference reuses the original Ray-free factory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from research import evaluate_pv_graph_ppo as graph_eval
from sep_research.checkpoints import fingerprint, _numeric_weights

SCHEMA = 'sep_legacy_inference_v1'
PUBLISHED = {
    'G_old': ('pv_graph_ppo/mixed_big_h128_l3_20m', 'periodic_iter_001100_steps_000018022400', 'graph'),
    'M_old': ('pv_nongraph_ppo/mixed_node_mlp_matched_20m', 'periodic_iter_000900_steps_000014745600', 'node-mlp-matched'),
    'C_old': ('pv_nongraph_ppo/mixed_grid_cnn_matched_20m', 'periodic_iter_001000_steps_000016085760', 'grid-cnn-matched'),
}
ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def checkpoint_hashes(path):
    files = {}
    for item in sorted(Path(path).rglob('*')):
        if item.is_symlink():
            raise ValueError('legacy checkpoint symlink')
        if item.is_file():
            files[str(item.relative_to(path))] = sha256(item)
    if not files or not (Path(path) / 'algorithm_state.pkl').is_file():
        raise ValueError('missing explicit legacy checkpoint binary')
    return files


def legacy_inputs(plan):
    if set(plan) != set(PUBLISHED):
        raise ValueError('explicit G_old, M_old, C_old checkpoints and run configs required')
    result = {}
    for method, (parent, name, kind) in PUBLISHED.items():
        entry = plan[method]
        if set(entry) != {'checkpoint', 'run_config'}:
            raise ValueError('each legacy input requires checkpoint and run_config only')
        path, config_path = Path(entry['checkpoint']).resolve(), Path(entry['run_config']).resolve()
        expected = ROOT / 'artifacts/rl' / parent / 'checkpoints' / name
        if path != expected or config_path != expected.parent.parent / 'run_config.json':
            raise ValueError(f'{method}: only the explicit published checkpoint/run config is accepted')
        run = json.loads(config_path.read_text())
        config = dict(graph_eval.DEFAULT_ENV_CONFIG)
        config.update({k: v for k, v in run.items() if k in config})
        config.update(run.get('env_config', {}))
        config.update(hidden_dim=run.get('hidden_dim', 64), message_layers=run.get('message_layers', 2))
        if kind != 'graph':
            config['model_preset'] = run.get('model_preset', run.get('model_spec', {}).get('preset'))
            if config['model_preset'] != kind:
                raise ValueError('legacy model kind mismatch')
        elif (config['hidden_dim'], config['message_layers']) != (128, 3):
            raise ValueError('published graph must be h128/l3')
        result[method] = dict(checkpoint=str(path), run_config_path=str(config_path),
            checkpoint_files=checkpoint_hashes(path), run_config_sha256=sha256(config_path),
            run_config=run, config=config, model_kind=kind)
    return result


def _mlp(left, hidden, right):
    return nn.Sequential(nn.Linear(left, hidden), nn.ReLU(), nn.Linear(hidden, right))


class _NodeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.message = nn.Sequential(*_mlp(128, 128, 128), nn.ReLU())
        self.update = _mlp(256, 128, 128)
        self.norm = nn.LayerNorm(128)

    def forward(self, nodes, mask):
        return F.relu(self.norm(nodes + self.update(torch.cat((nodes, self.message(nodes)), -1)))) * mask.unsqueeze(-1)


class _GridLayer(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.spatial = nn.Conv2d(80, 80, 3, dilation=(1, dilation), padding=(1, dilation))
        self.spatial_norm = nn.GroupNorm(8, 80)
        self.channel = nn.Conv2d(80, 80, 1)
        self.channel_norm = nn.GroupNorm(8, 80)

    def forward(self, grid):
        update = F.relu(self.spatial_norm(self.spatial(grid)))
        return F.relu(grid + self.channel_norm(self.channel(update)))


class _NonGraph(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        grid = kind == 'grid-cnn-matched'
        hidden = 80 if grid else 128
        self.global_encoder = _mlp(4, hidden, hidden)
        self.actor_head = _mlp(hidden if grid else 256, hidden, 3)
        self.value_head = _mlp(2 * hidden, hidden, 1)
        if grid:
            self.stem = nn.Conv2d(49, 80, 1)
            self.stem_norm = nn.GroupNorm(8, 80)
            self.residual_layers = nn.ModuleList(_GridLayer(d) for d in (1, 2, 4, 8, 16))
        else:
            self.node_encoder = _mlp(48, 128, 128)
            self.self_layers = nn.ModuleList(_NodeLayer() for _ in range(3))

    def forward(self, obs):
        features, mask = obs['node_features'].float(), obs['node_mask'].float()
        glob = self.global_encoder(obs['global_features'].float())
        if self.kind == 'node-mlp-matched':
            nodes = F.relu(self.node_encoder(features) + glob.unsqueeze(1)) * mask.unsqueeze(-1)
            for layer in self.self_layers:
                nodes = layer(nodes, mask)
            actor = torch.cat((nodes, glob.unsqueeze(1).expand(-1, 164, -1)), -1)
        else:
            relative = features[..., 0]
            column = (torch.where(relative >= 0, torch.round(relative * 30),
                                  torch.round(relative * 10)).long() + 10).clamp(0, 40)
            lane = torch.round(features[..., 1] * 3).long().clamp(0, 3)
            raster = features.new_zeros(features.shape[0], 4, 41, 49)
            b, s = torch.nonzero(mask.bool(), as_tuple=True)
            raster[b, lane[b, s], column[b, s], :48] = features[b, s]
            raster[b, lane[b, s], column[b, s], 48] = 1
            grid = self.stem_norm(self.stem(raster.permute(0, 3, 1, 2).contiguous()))
            grid = F.relu(grid + glob.unsqueeze(-1).unsqueeze(-1))
            for layer in self.residual_layers:
                grid = layer(grid)
            index = torch.arange(features.shape[0]).unsqueeze(1).expand_as(lane)
            nodes = grid.permute(0, 2, 3, 1)[index, lane, column] * mask.unsqueeze(-1)
            actor = nodes
        logits = self.actor_head(actor).masked_fill(~obs['action_mask'].bool(), -1e9)
        pooled = (nodes * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        value = self.value_head(torch.cat((pooled, glob), -1)).squeeze(-1)
        return logits, value


def build_legacy(weights, entry):
    weights = _numeric_weights(weights)
    config, kind = entry['config'], entry['model_kind']
    if (config['num_lanes'], config['front_distance'], config['back_distance']) != (4, 30, 10):
        raise ValueError('published legacy observation geometry required')
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        if kind == 'graph':
            model = graph_eval._make_inference_model(weights, config)
        elif kind in ('node-mlp-matched', 'grid-cnn-matched'):
            model = _NonGraph(kind)
            model.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()}, strict=True)
        else:
            raise ValueError('unknown legacy model kind')
    return model.eval().requires_grad_(False)


@torch.inference_mode()
def legacy_prediction(model, observation):
    tensors = {k: torch.as_tensor(v).unsqueeze(0) for k, v in observation.items()}
    out = model(tensors)
    logits, value = out[:2] if isinstance(out, tuple) else (out, None)
    return logits[0], None if value is None else value[0], logits[0].argmax(-1).numpy()


def load_legacy(path):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    if path.is_symlink() or '.partial_' in path.name or manifest.get('schema') != SCHEMA:
        raise ValueError('unpublished or invalid legacy cache')
    entries = manifest['models']
    current = legacy_inputs({m: dict(checkpoint=e['checkpoint'], run_config=e['run_config_path'])
                             for m, e in entries.items()})
    if set(p.name for p in path.iterdir()) != {'manifest.json', *(m + '.npz' for m in PUBLISHED)}:
        raise ValueError('missing or orphan legacy cache files')
    models = {}
    for method, entry in entries.items():
        if any(entry[k] != v for k, v in current[method].items()):
            raise ValueError('legacy checkpoint/config provenance mismatch')
        file = path / (method + '.npz')
        if file.is_symlink() or sha256(file) != entry['weights_sha256']:
            raise ValueError('legacy weight file hash mismatch')
        with np.load(file, allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)):
                raise ValueError('duplicate weight keys')
            weights = _numeric_weights(dict(archive))
        if fingerprint(weights) != entry['weights_fingerprint'] or entry['parity']['status'] != 'passed':
            raise ValueError('legacy weight identity/parity mismatch')
        models[method] = build_legacy(weights, entry)
    return models, manifest


def _restore_one(method, entry, directory):
    # Executed only in a disposable export subprocess; original helpers own Ray.
    import ray
    from research.nograph import evaluate_pv_nongraph_ppo as nongraph
    torch.set_num_threads(1)
    helper = graph_eval if method == 'G_old' else nongraph
    weights, smoke = helper._restore_policy(Path(entry['checkpoint']), entry['run_config'], entry['config'])
    original = helper._make_inference_model(weights, entry['config'])
    extracted = build_legacy(weights, entry)
    from snfs_traffic.rl.centralized_pv_env import CentralizedPvEnv
    env = CentralizedPvEnv(dict(entry['config'], road_length=100, warmup_steps=2, episode_steps=3))
    try:
        for fraction in (0., .95, 1.):
            obs, _ = env.reset(seed=810017, options={'av_fraction': fraction})
            for _ in range(3):
                tensors = {k: torch.as_tensor(v).unsqueeze(0) for k, v in obs.items()}
                with torch.inference_mode():
                    expected = original(tensors)
                logits, value, action = legacy_prediction(extracted, obs)
                old_logits = expected[0] if isinstance(expected, tuple) else expected
                torch.testing.assert_close(logits, old_logits[0], atol=1e-6, rtol=1e-6)
                np.testing.assert_array_equal(action, old_logits[0].argmax(-1).numpy())
                if isinstance(expected, tuple):
                    torch.testing.assert_close(value, expected[1][0], atol=1e-6, rtol=1e-6)
                obs, *_ = env.step(action)
    finally:
        env.close()
        if ray.is_initialized():
            ray.shutdown()
    np.savez(Path(directory) / (method + '.npz'), **_numeric_weights(weights))
    write_json(Path(directory) / (method + '.json'), dict(smoke, numeric_cases=9,
        logits_atol=1e-6, logits_rtol=1e-6, weights_fingerprint=fingerprint(weights)))


def export_legacy(plan, out_dir):
    """Export all three explicit trusted-local published checkpoints, exclusively."""
    inputs = legacy_inputs(plan)
    out = Path(out_dir).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{out.name}.partial_', dir=out.parent) as directory:
        temp = Path(directory)
        for method, entry in inputs.items():
            write_json(temp / 'input.json', entry)
            # A short private temp root avoids AF_UNIX limits; no shared Ray stop.
            with tempfile.TemporaryDirectory(prefix='sr_', dir='/tmp', ignore_cleanup_errors=True) as ray_temp:
                env = dict(os.environ, RAY_TMPDIR=ray_temp, RAY_ADDRESS='local', CUDA_VISIBLE_DEVICES='')
                subprocess.run([sys.executable, '-m', 'sep_research.export_legacy',
                    '--restore-one', method, '--plan', str(temp / 'input.json'), '--out-dir', str(temp)],
                    env=env, check=True)
            parity = json.loads((temp / (method + '.json')).read_text())
            entry.update(weights_sha256=sha256(temp / (method + '.npz')),
                         weights_fingerprint=parity.pop('weights_fingerprint'), parity=parity)
            (temp / (method + '.json')).unlink()
        (temp / 'input.json').unlink()
        write_json(temp / 'manifest.json', dict(schema=SCHEMA, models=inputs))
        out.mkdir()  # Exclusive publication reservation; failed reservations are invalid.
        temp.rename(out)
    load_legacy(out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--restore-one', choices=PUBLISHED, help=argparse.SUPPRESS)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.restore_one:
        _restore_one(args.restore_one, plan, args.out_dir)
    else:
        export_legacy(plan, args.out_dir)


if __name__ == '__main__':
    main()
