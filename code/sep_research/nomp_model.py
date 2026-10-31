"""Ray-free PV relation-set encoder; no simulator or training-side imports.

The input is the unchanged J_G PV observation, as batched tensors. Directed
relation slots (including repeated pairs) are encoded independently and pooled
in one global masked mean. Node representations are never updated by relations.
The graph's AV action mask is present but unused, as in the original PV encoder.
"""

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


GRAPH_SHAPES = {
    "node_features": (164, 48),
    "node_mask": (164,),
    "neighbor_index": (164, 6),
    "neighbor_mask": (164, 6),
    "edge_features": (164, 6, 3),
    "action_mask": (164, 3),
    "global_features": (4,),
}


class PvNoMPNetwork(nn.Module):
    """Fixed 411,572-parameter PV network; forward(obs) -> (logits[B,3], V[B]).

    Features are converted to float32, matching the article models. Masks and
    indices must follow the observation contract. Masked padding may contain
    arbitrary values, including invalid indices; an active out-of-range index
    fails rather than being clamped or silently removed. No inputs are mutated.
    """

    pv_encoder = "relation_set_mlp"
    graph_max_nodes = 164
    max_nodes = 1  # Factorized PPO loss axis, not the graph padding capacity.
    action_count = 3

    def __init__(self):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(48, 128), nn.ReLU(), nn.Linear(128, 128))
        self.global_encoder = nn.Sequential(
            nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 128))
        self.ego_encoder = nn.Sequential(
            nn.Linear(48, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.relation_encoder = nn.Sequential(
            nn.Linear(259, 240), nn.ReLU(), nn.Linear(240, 240), nn.ReLU(),
            nn.Linear(240, 128))
        self.fusion = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 128))
        self.actor_head = nn.Sequential(
            nn.Linear(384, 128), nn.ReLU(), nn.Linear(128, 3))
        self.value_head = nn.Sequential(
            nn.Linear(384, 128), nn.ReLU(), nn.Linear(128, 1))

    @staticmethod
    def _validate_shapes(obs):
        if not isinstance(obs, Mapping) or set(obs) != {
                "ego_features", "action_mask", "av_graph"}:
            raise ValueError("NoMP requires the unchanged J_G PV observation")
        ego = obs["ego_features"]
        if not isinstance(ego, torch.Tensor) or ego.ndim != 2 or ego.shape[1] != 48:
            raise ValueError("ego_features must be a batched tensor with shape (B, 48)")
        batch = ego.shape[0]
        if batch < 1 or tuple(obs["action_mask"].shape) != (batch, 3):
            raise ValueError("PV action_mask must have shape (B, 3), with B >= 1")
        graph = obs["av_graph"]
        if not isinstance(graph, Mapping) or set(graph) != set(GRAPH_SHAPES):
            raise ValueError("NoMP graph observation keys do not match J_G")
        for key, shape in GRAPH_SHAPES.items():
            if not isinstance(graph[key], torch.Tensor) or tuple(graph[key].shape) != (batch, *shape):
                raise ValueError(f"{key} must be a batched tensor with shape {(batch, *shape)}")

    def encode_graph(self, graph):
        """Return unchanged node embeddings, fused pooled context, global embedding."""
        node_mask = graph["node_mask"].bool()
        # Mask before Linear as well: padded NaN/Inf must not poison gradients.
        features = torch.where(node_mask.unsqueeze(-1),
                               graph["node_features"].float(), 0.0)
        global_embedding = self.global_encoder(graph["global_features"].float())
        nodes = F.relu(self.node_encoder(features) + global_embedding.unsqueeze(1))
        nodes = torch.where(node_mask.unsqueeze(-1), nodes, 0.0)
        node_pool = nodes.sum(1) / node_mask.sum(1, keepdim=True).clamp_min(1)

        slots = node_mask.unsqueeze(-1) & graph["neighbor_mask"].bool()
        indices = torch.where(slots, graph["neighbor_index"].long(), 0)
        batch, count, neighbors = indices.shape
        flat_indices = indices.reshape(batch, count * neighbors)
        # gather rejects both negative and positive OOB indices on active slots.
        target_mask = node_mask.gather(1, flat_indices).reshape(batch, count, neighbors)
        relation_mask = slots & target_mask
        targets = nodes.gather(
            1, flat_indices.unsqueeze(-1).expand(-1, -1, 128)
        ).reshape(batch, count, neighbors, 128)
        sources = nodes.unsqueeze(2).expand(-1, -1, neighbors, -1)
        edges = torch.where(relation_mask.unsqueeze(-1),
                            graph["edge_features"].float(), 0.0)
        relations = self.relation_encoder(torch.cat((sources, targets, edges), dim=-1))
        relations = torch.where(relation_mask.unsqueeze(-1), relations, 0.0)
        denominator = relation_mask.sum((1, 2)).clamp_min(1).unsqueeze(-1)
        relation_pool = relations.sum((1, 2)) / denominator
        pooled = self.fusion(torch.cat((node_pool, relation_pool), dim=-1))
        return nodes, pooled, global_embedding

    def forward(self, obs):
        self._validate_shapes(obs)
        _, pooled, global_embedding = self.encode_graph(obs["av_graph"])
        context = torch.cat((self.ego_encoder(obs["ego_features"].float()),
                             pooled, global_embedding), dim=-1)
        logits = self.actor_head(context).masked_fill(~obs["action_mask"].bool(), -1.0e9)
        return logits, self.value_head(context).squeeze(-1)
