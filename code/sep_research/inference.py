"""CPU float32 inference from already-loaded, trusted policy state dictionaries.

No environment stepping. load_pair reads validated numeric checkpoint exports. PV encoders are exactly ``local_mlp``
(23,812 parameters) and ``graph_mlp`` (411,012); AV is h128/3MP (355,460),
including legacy AV weights with that architecture. State keys are unchanged.
Observations are the unflattened training dictionaries (NumPy arrays or tensors).
The caller may reuse one raw AV graph in both observations; no inputs are changed.
"""

from collections.abc import Mapping
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from research.evaluate_pv_graph_ppo import DEFAULT_ENV_CONFIG, _make_inference_model


class Prediction(NamedTuple):
    """CPU tensors: masked logits, calculated value, and argmax action indices.

    Single PV: [3], [], []; single AV: [164,3], [], [164]. An explicitly
    batched observation adds a leading B to each output, even when B=1.
    Ties choose the first index, as in torch.argmax; indices map to left/stay/right.
    """

    logits: torch.Tensor
    value: torch.Tensor
    action: torch.Tensor


class _MessagePassingLayer(nn.Module):
    # Mirrored from snfs_traffic.rl.graph_ppo._MessagePassingLayer to avoid Ray.
    def __init__(self):
        super().__init__()
        self.message = nn.Sequential(nn.Linear(131, 128), nn.ReLU(),
                                     nn.Linear(128, 128), nn.ReLU())
        self.update = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 128))
        self.norm = nn.LayerNorm(128)

    def forward(self, nodes, neighbor_index, neighbor_mask, edge_features, node_mask):
        batch, count, neighbors = neighbor_index.shape
        index = torch.arange(batch, device=nodes.device).view(batch, 1, 1)
        index = index.expand(batch, count, neighbors)
        messages = self.message(torch.cat((nodes[index, neighbor_index], edge_features), -1))
        weights = neighbor_mask.unsqueeze(-1)
        messages = messages * weights
        pooled = messages.sum(2) / weights.sum(2).clamp_min(1.0)
        update = self.update(torch.cat((nodes, pooled), -1))
        return F.relu(self.norm(nodes + update)) * node_mask.unsqueeze(-1)


def _encode_graph(model, graph):
    """Shared AV/PV computation, using only the supplied model's own parameters."""
    mask = graph["node_mask"].float()
    global_embedding = model.global_encoder(graph["global_features"].float())
    nodes = F.relu(model.node_encoder(graph["node_features"].float())
                   + global_embedding.unsqueeze(1)) * mask.unsqueeze(-1)
    for layer in model.message_layers:
        nodes = layer(nodes, graph["neighbor_index"].long(), graph["neighbor_mask"].float(),
                      graph["edge_features"].float(), mask)
    pooled = (nodes * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
    return nodes, pooled, global_embedding


class _PvModel(nn.Module):
    # Mirrored from sep_research.models; preserve registration names and concat order.
    def __init__(self, pv_encoder):
        super().__init__()
        self.pv_encoder = pv_encoder
        graph = pv_encoder == "graph_mlp"
        if graph:
            self.node_encoder = nn.Sequential(nn.Linear(48, 128), nn.ReLU(), nn.Linear(128, 128))
            self.global_encoder = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 128))
            self.message_layers = nn.ModuleList(_MessagePassingLayer() for _ in range(3))
        self.ego_encoder = nn.Sequential(nn.Linear(48 if graph else 52, 128), nn.ReLU(),
                                         nn.Linear(128, 128), nn.ReLU())
        self.actor_head = (nn.Sequential(nn.Linear(384, 128), nn.ReLU(), nn.Linear(128, 3))
                           if graph else nn.Linear(128, 3))
        self.value_head = (nn.Sequential(nn.Linear(384, 128), nn.ReLU(), nn.Linear(128, 1))
                           if graph else nn.Linear(128, 1))

    def forward(self, obs):
        ego = obs["ego_features"].float()
        if self.pv_encoder == "graph_mlp":
            _, pooled, global_embedding = _encode_graph(self, obs["av_graph"])
            context = torch.cat((self.ego_encoder(ego), pooled, global_embedding), -1)
        else:
            context = self.ego_encoder(torch.cat((ego, obs["global_features"].float()), -1))
        logits = self.actor_head(context).masked_fill(~obs["action_mask"].bool(), -1e9)
        return logits, self.value_head(context).squeeze(-1)


class _AvModel(nn.Module):
    def __init__(self, legacy):
        super().__init__()
        self.max_nodes = legacy.max_nodes
        # Register the factory's modules directly: no wrapper prefix in state_dict.
        for name, module in legacy.named_children():
            self.add_module(name, module)

    def forward(self, obs):
        nodes, pooled, global_embedding = _encode_graph(self, obs)
        repeated = global_embedding.unsqueeze(1).expand(-1, self.max_nodes, -1)
        logits = self.actor_head(torch.cat((nodes, repeated), -1))
        logits = logits.masked_fill(~obs["action_mask"].bool(), -1e9)
        value = self.value_head(torch.cat((pooled, global_embedding), -1)).squeeze(-1)
        return logits, value


def _weights(state_dict):
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("state_dict must be a nonempty mapping of parameter names to arrays/tensors")
    tensors = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise ValueError("state_dict parameter names must be strings")
        try:
            tensor = torch.as_tensor(value).detach().cpu()
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"invalid weight {key}: expected a numeric array/tensor") from exc
        if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise ValueError(f"invalid weight {key}: expected finite floating-point values")
        tensor = tensor.float()
        if not torch.isfinite(tensor).all():
            raise ValueError(f"invalid weight {key}: overflows float32")
        tensors[key] = tensor
    return tensors


def build_pv_model(state_dict, pv_encoder):
    """Strictly load independent, frozen CPU PV weights; preserve caller CPU RNG.

    ``pv_encoder`` must be local_mlp or graph_mlp; mismatched keys/shapes raise
    RuntimeError from strict loading, invalid encoders/weight values ValueError.
    No parameters alias the input state_dict or any previously built model.
    """
    if pv_encoder not in ("local_mlp", "graph_mlp"):
        raise ValueError("pv_encoder must be 'local_mlp' or 'graph_mlp'")
    weights = _weights(state_dict)
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = _PvModel(pv_encoder).float()
    model.load_state_dict(weights, strict=True)
    return model.eval().requires_grad_(False)


def build_av_model(state_dict, graph_config):
    """Load fixed h128/3MP AV weights using the existing Ray-free legacy factory.

    graph_config is an environment-config mapping, merged with legacy defaults
    (four lanes, window [-10,+30]). Capacity must be 164. Optional hidden_dim and
    message_layers must be 128 and 3. No simulation is constructed or stepped.
    Strict loading and independent parameter ownership match build_pv_model.
    """
    config = dict(DEFAULT_ENV_CONFIG, hidden_dim=128, message_layers=3)
    config.update(graph_config)
    if config["hidden_dim"] != 128 or config["message_layers"] != 3:
        raise ValueError("AV inference requires hidden_dim=128 and message_layers=3")
    capacity = config["num_lanes"] * (config["front_distance"] + config["back_distance"] + 1)
    if capacity != 164:
        raise ValueError("AV inference requires graph capacity 164")
    weights = {key: value.numpy() for key, value in _weights(state_dict).items()}
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = _AvModel(_make_inference_model(weights, config)).float()
    return model.eval().requires_grad_(False)


_GRAPH_SHAPES = {"node_features": (164, 48), "node_mask": (164,),
                 "neighbor_index": (164, 6), "neighbor_mask": (164, 6),
                 "edge_features": (164, 6, 3), "action_mask": (164, 3),
                 "global_features": (4,)}


def _tensor_observation(obs, shapes, batch):
    if not isinstance(obs, Mapping) or set(obs) != set(shapes):
        raise ValueError("observation keys do not match the selected model")
    result = {}
    for key, shape in shapes.items():
        if isinstance(shape, dict):
            result[key] = _tensor_observation(obs[key], shape, batch)
            continue
        tensor = torch.as_tensor(obs[key], device="cpu").detach()
        expected = shape if batch is None else (batch, *shape)
        if tuple(tensor.shape) != expected:
            raise ValueError(f"{key}: expected shape {expected}, got {tuple(tensor.shape)}")
        result[key] = tensor.unsqueeze(0) if batch is None else tensor
    return result


@torch.inference_mode()
def predict(model, observation):
    """Return Prediction for one observation or an already-stacked batch.

    Shapes and schema are checked; feature values/indices/masks are expected to
    follow the environment contract. Does not mutate inputs or consume RNG.
    """
    if isinstance(model, _PvModel):
        shapes = {"ego_features": (48,), "action_mask": (3,)}
        shapes.update({"av_graph": _GRAPH_SHAPES} if model.pv_encoder == "graph_mlp"
                      else {"global_features": (4,)})
        key, single_rank = "ego_features", 1
    elif isinstance(model, _AvModel):
        shapes = _GRAPH_SHAPES
        key, single_rank = "node_features", 2
    else:
        raise TypeError("model must come from build_pv_model or build_av_model")
    feature = torch.as_tensor(observation[key])
    batch = None if feature.ndim == single_rank else feature.shape[0]
    tensors = _tensor_observation(observation, shapes, batch)
    logits, value = model(tensors)
    if batch is None:
        logits, value = logits[0], value[0]
    return Prediction(logits, value, logits.argmax(dim=-1))


def load_pair(bundle, expected):
    """Return (CPU models by policy ID, manifest), with native roles None.

    Supply the original frozen training contract. Trusted-local integrity checks
    do not authenticate externally supplied manifests or models.
    """
    from sep_research.checkpoints import load_inference_weights
    weights, manifest = load_inference_weights(bundle, expected)
    models = {}
    for pid, state in weights.items():
        models[pid] = (None if state is None else
                       build_pv_model(state, manifest['pv_encoder']) if pid == 'pv_policy'
                       else build_av_model(state, manifest['contract']['env']))
    return models, manifest
