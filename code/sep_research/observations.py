"""Pure PV ego features, following snfs_traffic.observations.pv_graph.

The small ego-only duplication preserves the article AV builder unchanged.
Distances are directed head-cell distances, including its ID tie breaking.
"""

import numpy as np

from snfs_traffic.control import _lateral_valid
from snfs_traffic.core import build_body_occupancy, build_lane_order, build_occupancy
from snfs_traffic.observations.pv_graph import _nearest_by_direction, _type_flags
from snfs_traffic.scenarios import HDV_VEH_TYPE, PRIORITY_VEH_TYPE


def build_pv_ego_observation(state, pv_id, params, graph_config, cooldowns):
    """Return ego_features[48] and action_mask[3], without changing inputs."""
    matches = np.flatnonzero(state.alive & (state.vehicle_id == pv_id))
    priorities = np.flatnonzero(state.alive & (state.veh_type == PRIORITY_VEH_TYPE))
    if matches.size != 1 or not np.array_equal(matches, priorities):
        raise ValueError("PV must be the unique alive priority vehicle")
    graph_config.validate_for(params)
    idx = int(matches[0])
    lane, speed = int(state.lane[idx]), int(state.vel[idx])
    lane_scale = max(params.num_lanes - 1, 1)
    max_speed = max(params.vmax_default, params.vmax_controlled, 1)
    remaining = max(0, int(cooldowns.get(int(pv_id), 0)))
    features = np.zeros(48, dtype=np.float32)
    features[1] = lane / lane_scale
    features[3] = speed / max_speed
    features[5] = state.last_lane_delta[idx]
    features[6] = remaining / max(graph_config.cooldown_steps, 1)
    alive = np.flatnonzero(state.alive)
    for block, delta in enumerate((-1, 0, 1)):
        target_lane = lane + delta
        if not 0 <= target_lane < params.num_lanes:
            continue
        cursor = 7 + block * 11
        features[cursor] = 1
        candidates = alive[state.lane[alive] == target_lane]
        for direction, offset in (("front", 1), ("back", 6)):
            neighbor, distance = _nearest_by_direction(
                state, ego_idx=idx, candidate_indices=candidates,
                direction=direction, road_length=params.road_length,
            )
            if neighbor is not None and distance <= graph_config.sensor_distance:
                hdv, pv = _type_flags(state, neighbor, idx)
                features[cursor + offset:cursor + offset + 5] = (
                    1, distance / graph_config.sensor_distance,
                    (int(state.vel[neighbor]) - speed) / max_speed, hdv, pv,
                )
    hdvs = alive[state.veh_type[alive] == HDV_VEH_TYPE]
    for direction, cursor in (("front", 40), ("back", 44)):
        neighbor, distance = _nearest_by_direction(
            state, ego_idx=idx, candidate_indices=hdvs,
            direction=direction, road_length=params.road_length,
        )
        if neighbor is not None and distance <= graph_config.sensor_distance:
            features[cursor:cursor + 4] = (
                1, distance / graph_config.sensor_distance,
                (int(state.lane[neighbor]) - lane) / lane_scale,
                (int(state.vel[neighbor]) - speed) / max_speed,
            )
    mask = np.asarray([0, 1, 0], dtype=np.int8)
    if remaining == 0:
        occupancy = build_occupancy(state, params)
        bodies = build_body_occupancy(state, params)
        order, counts, _ = build_lane_order(occupancy, n_vehicles=state.n_vehicles)
        for column, delta in ((0, -1), (2, 1)):
            if 0 <= lane + delta < params.num_lanes:
                valid, _ = _lateral_valid(
                    state, order, counts, occupancy, bodies, idx,
                    lane + delta, params.road_length,
                )
                mask[column] = int(valid)
    return {"ego_features": features, "action_mask": mask}
