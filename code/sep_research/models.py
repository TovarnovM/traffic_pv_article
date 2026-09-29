"""Independent PV models for the existing FactorizedPPOTorchPolicy (N=1)."""

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from snfs_traffic.rl.graph_ppo import _MessagePassingLayer


def _ego_encoder(inputs):
    return nn.Sequential(nn.Linear(inputs, 128), nn.ReLU(),
                         nn.Linear(128, 128), nn.ReLU())


class PvManeuverTorchModel(TorchModelV2, nn.Module):
    """Local PV: ego48 + global4, fixed article milestone architecture."""

    graph_encoder = False

    def __init__(self, obs_space, action_space, num_outputs, model_config, name,
                 **custom_options):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        if num_outputs != 3:
            raise ValueError("PV model requires exactly three logits")
        if custom_options or model_config.get("custom_model_config"):
            raise ValueError("PV milestone architecture has no custom options")
        original = getattr(obs_space, "original_space", obs_space)
        expected = {"ego_features", "action_mask",
                    "av_graph" if self.graph_encoder else "global_features"}
        if set(original.spaces) != expected or original["ego_features"].shape != (48,):
            raise ValueError("observation schema does not match PV encoder")
        if original["action_mask"].shape != (3,):
            raise ValueError("PV action mask must have shape (3,)")
        self.max_nodes = 1  # PPO loss axis, never graph padding capacity.
        self.action_count = 3
        self._last_value = None
        self._last_node_mask = None
        if self.graph_encoder:
            graph_space = original["av_graph"]
            self.graph_max_nodes = graph_space["node_features"].shape[0]
            self.node_encoder = nn.Sequential(
                nn.Linear(48, 128), nn.ReLU(), nn.Linear(128, 128))
            self.global_encoder = nn.Sequential(
                nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 128))
            self.message_layers = nn.ModuleList(_MessagePassingLayer(128, 3) for _ in range(3))
            self.ego_encoder = _ego_encoder(48)
            self.actor_head = nn.Sequential(nn.Linear(384, 128), nn.ReLU(), nn.Linear(128, 3))
            self.value_head = nn.Sequential(nn.Linear(384, 128), nn.ReLU(), nn.Linear(128, 1))
        else:
            self.ego_encoder = _ego_encoder(52)
            self.actor_head = nn.Linear(128, 3)
            self.value_head = nn.Linear(128, 1)

    def encode_graph(self, graph):
        """Article-identical node, masked pool and global embeddings."""
        mask = graph["node_mask"].float()
        global_embedding = self.global_encoder(graph["global_features"].float())
        nodes = F.relu(self.node_encoder(graph["node_features"].float())
                       + global_embedding.unsqueeze(1)) * mask.unsqueeze(-1)
        for layer in self.message_layers:
            nodes = layer(nodes, graph["neighbor_index"].long(),
                          graph["neighbor_mask"].float(), graph["edge_features"].float(), mask)
        pooled = (nodes * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return nodes, pooled, global_embedding

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        if not isinstance(obs, Mapping):
            obs = restore_original_dimensions(obs, self.obs_space, "torch")
        ego = obs["ego_features"].float()
        if self.graph_encoder:
            _, pooled, global_embedding = self.encode_graph(obs["av_graph"])
            context = torch.cat((self.ego_encoder(ego), pooled, global_embedding), dim=-1)
        else:
            context = self.ego_encoder(torch.cat((ego, obs["global_features"].float()), dim=-1))
        logits = self.actor_head(context).masked_fill(~obs["action_mask"].bool(), -1.0e9)
        self._last_value = self.value_head(context).squeeze(-1)
        self._last_node_mask = logits.new_ones((logits.shape[0], 1))
        return logits, state

    @property
    def last_node_mask(self):
        if self._last_node_mask is None:
            raise RuntimeError("forward must run before reading node mask")
        return self._last_node_mask

    def value_function(self):
        if self._last_value is None:
            raise RuntimeError("forward must run before value_function")
        return self._last_value


class PvGraphManeuverTorchModel(PvManeuverTorchModel):
    """PV ego branch plus an independent article AV graph backbone."""

    graph_encoder = True
