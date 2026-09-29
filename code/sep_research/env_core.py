"""Ray-free joint transition core for two persistent policy roles."""

from collections.abc import Mapping

import numpy as np
from gymnasium import spaces

from snfs_traffic.control import LateralOverrideResult
from snfs_traffic.core.invariants import validate_runtime_invariants
from snfs_traffic.rl.centralized_pv_env import (
    CentralizedPvEnv, PvSpeedRewardConfig, compute_pv_speed_reward, _condition_key,
)
from snfs_traffic.scenarios import AV_VEH_TYPE, PRIORITY_VEH_TYPE
from snfs_traffic.simulator import TrafficSimulator

from sep_research.observations import build_pv_ego_observation


class SepResearchCore:
    """Classic RLlib API, with IDs ``pv`` and ``av_coordinator``.

    ``controller_mode`` is A/P_L/P_G/J_L/J_G (default J_L or J_G selected by
    pv_encoder). Native roles remain observed/rewarded; optional sampler actions are
    validated but issue NO overrides. PV actions accept integer scalars (including
    zero-dimensional integer arrays); AV actions accept integer arrays or
    lists/tuples of integers. Booleans and floats are rejected. Frozen policies use the same action interface
    as trainable policies. ``action_agents`` tells runners which keys to send.

    All article config/reset options pass directly to CentralizedPvEnv,
    including EnvContext worker/vector attributes. For fixed evaluation use
    singleton density_values/av_fraction_values and warmup_cache=False.
    ``state`` is a copy; ``simulator`` exposes the single simulator for metrics.
    Private physical slot IDs live in ``_article._current_graph.vehicle_id``.
    """

    def __init__(self, config=None):
        super().__init__()
        config = config if config is not None else {}
        encoder = config.get("pv_encoder", "local_mlp")
        mode = config.get("controller_mode", "J_G" if encoder == "graph_mlp" else "J_L")
        if mode not in {"A", "P_L", "P_G", "J_L", "J_G"}:
            raise ValueError("controller_mode must be A/P_L/P_G/J_L/J_G")
        expected = "graph_mlp" if mode.endswith("_G") else "local_mlp"
        if mode != "A":
            if "pv_encoder" in config and encoder != expected:
                raise ValueError("pv_encoder conflicts with controller_mode")
            encoder = expected
        if encoder not in {"local_mlp", "graph_mlp"}:
            raise ValueError("pv_encoder must be local_mlp or graph_mlp")
        self.pv_encoder = encoder
        self.controller_mode = mode
        self.possible_agents = ["pv", "av_coordinator"]
        self.agents = self.possible_agents.copy()
        self._agent_ids = set(self.possible_agents)
        self.action_agents = tuple(
            agent for agent in self.possible_agents
            if (agent == "pv" and mode != "A")
            or (agent == "av_coordinator" and not mode.startswith("P_"))
        )
        if "seed_guard" in config:
            from sep_research.seed_control import GuardedCentralizedPvEnv
            self._article = GuardedCentralizedPvEnv(config)
        else:
            self._article = CentralizedPvEnv(config)
        if (self.params.vmax_default, self.params.vmax_controlled) != (5, 6):
            raise ValueError("article speed limits must be HDV/AV=5, PV=6")
        if self._article.reward_config != PvSpeedRewardConfig():
            raise ValueError("sep_research requires the original .001/.002 AV penalties")
        pv_spaces = {
            "ego_features": spaces.Box(-1, 1, (48,), dtype=np.float32),
            "action_mask": spaces.MultiBinary(3),
        }
        if encoder == "graph_mlp":
            pv_spaces["av_graph"] = self._article.observation_space
        else:
            pv_spaces["global_features"] = spaces.Box(-1, 1, (4,), dtype=np.float32)
        self.observation_spaces = {
            "pv": spaces.Dict(pv_spaces),
            "av_coordinator": self._article.observation_space,
        }
        self.action_spaces = {
            "pv": spaces.Discrete(3), "av_coordinator": self._article.action_space,
        }
        self.observation_space = spaces.Dict(self.observation_spaces)
        self.action_space = spaces.Dict(self.action_spaces)
        self._reset_info = {}
        self._team_return = 0.0

    @property
    def params(self):
        return self._article.params

    @property
    def simulator(self):
        return self._article._sim

    @property
    def state(self):
        return self.simulator.state

    @property
    def priority_vehicle_id(self):
        return self._article.priority_vehicle_id

    @property
    def backend_name(self):
        return self.simulator.backend_name

    def get_observation_space(self, agent_id):
        return self.observation_spaces[agent_id]

    def get_action_space(self, agent_id):
        return self.action_spaces[agent_id]

    def close(self):
        self._article.close()

    def _validate_state(self, state, priority_vehicle_id=None):
        validate_runtime_invariants(state, self.params, self.simulator.topology)
        if priority_vehicle_id is None:
            priority_vehicle_id = self.priority_vehicle_id
        matches = np.flatnonzero(state.alive & (state.vehicle_id == priority_vehicle_id))
        priorities = np.flatnonzero(state.alive & (state.veh_type == PRIORITY_VEH_TYPE))
        if matches.size != 1 or not np.array_equal(matches, priorities):
            raise ValueError("invalid episode: PV must be the unique alive priority vehicle")
        if np.any(state.controlled):
            raise ValueError("invalid episode: all controlled flags must be False")

    def _observations(self, state, av_graph=None):
        self._validate_state(state)
        if av_graph is None:
            av_graph = self._article._build_observation(state)
        pv = build_pv_ego_observation(
            state, self.priority_vehicle_id, self.params,
            self._article.graph_config, self._article._cooldown_by_vehicle_id,
        )
        if self.pv_encoder == "graph_mlp":
            pv["av_graph"] = av_graph
        else:
            pv["global_features"] = av_graph["global_features"]
        return {"pv": pv, "av_coordinator": av_graph}

    def get_seed_audit(self):
        """Copy of every guarded training reset, including the live episode."""
        if not hasattr(self._article, 'get_seed_audit'):
            raise ValueError('training seed guard is not enabled')
        return self._article.get_seed_audit()

    def reset(self, *, seed=None, options=None):
        av_graph, info = self._article.reset(seed=seed, options=options)
        try:
            return self._reset_output(av_graph, info)
        except BaseException:
            if hasattr(self._article, '_seed_audit'):
                self._article._seed_audit['failed_resets'] += 1
            raise

    def reset_from_snapshot(self, state, *, priority_vehicle_id, scenario_seed,
                            warmup_rng_seed, assignment_seed, av_selection_seed,
                            measurement_rng_seed, density, av_fraction, warmup_steps):
        """Install a copy of an externally warmed/assigned/tagged TrafficState.

        Required metadata describes actual provenance; callers attest creation-time
        AV tagging. No warmup, assignment, selection or measurement RNG draws occur.
        Returns (observations, per-role infos), as reset does. Training seed/grid/
        cache state is untouched; external_warmup_reuse=True, warmup_cache_hit=False.
        """
        seeds = dict(scenario_seed=scenario_seed, warmup_rng_seed=warmup_rng_seed,
                     assignment_seed=assignment_seed, av_selection_seed=av_selection_seed,
                     measurement_rng_seed=measurement_rng_seed)
        for name, value in dict(seeds, priority_vehicle_id=priority_vehicle_id,
                                warmup_steps=warmup_steps).items():
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer)) or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer")
        for name, value in (("density", density), ("av_fraction", av_fraction)):
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, float, np.integer, np.floating))
                    or not np.isfinite(value)
                    or not (0 < value < 1 if name == "density" else 0 <= value <= 1)):
                raise ValueError(f"invalid {name}")
        self._validate_state(state, priority_vehicle_id)
        owner = self._article
        simulator = TrafficSimulator(params=self.params, backend=owner._backend,
                                     rng_seed=int(measurement_rng_seed), validate=owner._validate)
        installed = simulator.reset(state=state, rng_seed=int(measurement_rng_seed))
        if hasattr(owner, '_training_measurement_active'):
            owner._training_measurement_active = False
        owner._sim = simulator
        owner._priority_vehicle_id = int(priority_vehicle_id)
        owner._cooldown_by_vehicle_id.clear()
        owner._step_index = 0
        owner._pv_speed_sum = 0.0
        owner._episode_applied_count = 0
        owner._episode_rejected_count = 0
        owner._done = False
        av_graph = owner._build_observation(installed)
        info = dict(
            seeds, step_index=0, priority_vehicle_id=self.priority_vehicle_id,
            active_av_count=owner._current_graph.active_count,
            fleet_av_count=int(np.sum(installed.alive & (installed.veh_type == AV_VEH_TYPE))),
            density=float(density), av_fraction=float(av_fraction),
            condition_key=_condition_key(density, av_fraction),
            condition_cycle=0, condition_count=1, warmup_steps=int(warmup_steps),
            warmup_cache_hit=False, warmup_cache_size=len(owner._warmup_cache),
            external_warmup_reuse=True, backend_name=simulator.backend_name,
        )
        return self._reset_output(av_graph, info)

    def _reset_output(self, av_graph, info):
        self.agents = self.possible_agents.copy()
        self._team_return = 0.0
        self._reset_info = dict(info, controller_mode=self.controller_mode,
                                pv_encoder=self.pv_encoder)
        observations = self._observations(self.state, av_graph)
        return observations, {agent: dict(self._reset_info, team_return=0.0)
                              for agent in self.agents}

    def _validate_actions(self, actions):
        if not isinstance(actions, Mapping) or set(actions) not in (
            set(self.action_agents), set(self.possible_agents)
        ):
            raise ValueError(f"expected active keys {self.action_agents} or both roles")
        arrays = {}
        for agent in actions:
            raw = actions[agent]
            if isinstance(raw, (list, tuple)) and any(
                isinstance(item, (bool, np.bool_))
                or not isinstance(item, (int, np.integer)) for item in raw
            ):
                raise ValueError(f"invalid integer action for {agent}")
            value = np.asarray(raw)
            if (value.dtype.kind not in "iu"
                    or value.shape != self.action_spaces[agent].shape
                    or np.any(value < 0) or np.any(value > 2)):
                raise ValueError(f"invalid integer action for {agent}")
            arrays[agent] = value
        return arrays

    def step(self, action_dict):
        owner = self._article
        if owner._done:
            raise RuntimeError("episode is done; call reset() before step()")
        actions = self._validate_actions(action_dict)
        graph = owner._current_graph
        commands = {}
        for agent in self.action_agents:
            ids = ([self.priority_vehicle_id] if agent == "pv"
                   else graph.vehicle_id[:graph.active_count])
            values = actions[agent].reshape(-1)
            commands[agent] = {int(vid): int(value) - 1 for vid, value in zip(ids, values)}
        return self.step_commands(commands)

    def step_commands(self, commands, *, apply_cooldowns=True):
        """Step role -> {physical ID: integer delta in -1/0/1} sparse commands.

        Only controller_mode's active roles are allowed. Missing IDs stay native;
        explicit zero forces stay. Empty mappings give native B1. B2 must use
        apply_cooldowns=False: applied cooldowns remain tracked for observations,
        but filtering belongs to its controller. Returns step's usual five-tuple
        and event ledger. Invalid commands fail before any episode/RNG mutation.
        """
        owner = self._article
        if owner._done:
            raise RuntimeError("episode is done; call reset() before step()")
        if not isinstance(apply_cooldowns, bool):
            raise ValueError("apply_cooldowns must be bool")
        if not isinstance(commands, Mapping) or not set(commands) <= set(self.action_agents):
            raise ValueError(f"commands require allowed roles {self.action_agents}")
        before = self.state
        self._validate_state(before)
        seen = set()
        for agent, mapping in commands.items():
            if not isinstance(mapping, Mapping):
                raise ValueError("role commands must be a mapping")
            for vid, delta in mapping.items():
                if any(isinstance(value, (bool, np.bool_))
                       or not isinstance(value, (int, np.integer)) for value in (vid, delta)):
                    raise ValueError("physical IDs and deltas must be integers")
                indices = np.flatnonzero(before.alive & (before.vehicle_id == vid))
                if indices.size != 1 or vid in seen or delta not in (-1, 0, 1):
                    raise ValueError("invalid or duplicate physical ID or delta")
                if ((agent == "pv" and vid != self.priority_vehicle_id)
                        or (agent == "av_coordinator" and before.veh_type[indices[0]] != AV_VEH_TYPE)):
                    raise ValueError("physical ID does not belong to role")
                seen.add(vid)
        graph = owner._current_graph
        events = {agent: dict(raw_nonstay=0, cooldown=0, issued=0,
                              applied=0, blocked=0, conflict=0, rejected=0)
                  for agent in self.possible_agents}
        for event in events.values():
            event.update(raw_left=0, raw_stay=0, raw_right=0,
                         applied_left=0, applied_right=0,
                         native_left=0, native_right=0)
        overrides = {}
        role_by_id = {}
        for agent in self.possible_agents:
            for vehicle_id, delta in commands.get(agent, {}).items():
                vehicle_id = int(vehicle_id)
                delta = int(delta)
                event = events[agent]
                event[("raw_left", "raw_stay", "raw_right")[delta + 1]] += 1
                if delta:
                    event["raw_nonstay"] += 1
                    if apply_cooldowns and owner._cooldown_by_vehicle_id.get(vehicle_id, 0) > 0:
                        event["cooldown"] += 1
                        event["rejected"] += 1
                        delta = 0
                    else:
                        event["issued"] += 1
                overrides[vehicle_id] = delta
                role_by_id[vehicle_id] = agent
        # Empty mappings also use the facade: it preserves the native RNG path.
        audit = (owner._seed_audit if getattr(owner, '_training_measurement_active', False)
                 else None)
        if audit is not None:
            audit['pending_transitions'] += 1
        state = self.simulator.step_lateral_overrides(overrides)
        if audit is not None:
            audit['measurement_steps'] += 1
            audit['episode_measurement_steps'] += 1
        result = self.simulator.last_step_info.action_result
        if not isinstance(result, LateralOverrideResult):
            owner._done = True
            raise RuntimeError("lateral transition returned no result")
        try:
            self._validate_state(state)
        except ValueError:
            owner._done = True
            raise
        for vehicle_id, delta, applied, reason in zip(
            result.vehicle_id, result.requested_lane_delta,
            result.applied, result.rejection_reason,
        ):
            if not delta:
                continue
            event = events[role_by_id[int(vehicle_id)]]
            if applied:
                event["applied"] += 1
                event["applied_left" if delta < 0 else "applied_right"] += 1
            else:
                event["rejected"] += 1
                event["conflict" if reason == "override_conflict" else "blocked"] += 1
        for idx in np.flatnonzero(state.alive & state.changed_lane):
            if int(state.vehicle_id[idx]) in overrides:
                continue
            if int(state.veh_type[idx]) == PRIORITY_VEH_TYPE:
                role = "pv"
            elif int(state.veh_type[idx]) == AV_VEH_TYPE:
                role = "av_coordinator"
            else:
                continue
            events[role]["native_left" if state.last_lane_delta[idx] < 0 else "native_right"] += 1
        owner._update_cooldowns(result)
        owner._step_index += 1
        pv_index = np.flatnonzero(state.alive & (state.vehicle_id == self.priority_vehicle_id))[0]
        speed = int(state.vel[pv_index])
        av = events["av_coordinator"]
        reward, components = compute_pv_speed_reward(
            pv_speed=speed, pv_vmax=6, applied_count=av["applied"],
            rejected_count=av["rejected"], config=owner.reward_config,
        )
        self._team_return += reward
        owner._pv_speed_sum += speed
        owner._episode_applied_count += av["applied"]
        owner._episode_rejected_count += av["rejected"]
        owner._done = owner._step_index >= owner._episode_steps
        observations = self._observations(state)
        info = dict(
            self._reset_info, step_index=owner._step_index, pv_speed=speed,
            pv_speed_norm=speed / 6, team_reward=reward, team_return=self._team_return,
            episode_mean_pv_speed=owner._pv_speed_sum / owner._step_index,
            active_av_count=graph.active_count,
            next_active_av_count=owner._current_graph.active_count,
            role_events=events,
            total_events={key: sum(event[key] for event in events.values())
                          for key in events["pv"]},
            reward_components=components,
            episode_applied_count=owner._episode_applied_count,
            episode_rejected_count=owner._episode_rejected_count,
            backend_name=self.simulator.last_step_info.backend_name,
            used_reference_action_path=self.simulator.last_step_info.used_reference_action_path,
        )
        if audit is not None:
            audit['pending_transitions'] -= 1
        return (
            observations, {agent: reward for agent in self.agents},
            {key: False for key in [*self.agents, "__all__"]},
            {key: owner._done for key in [*self.agents, "__all__"]},
            {agent: dict(info) for agent in self.agents},
        )
