"""Pure provenance contracts for EXPERIMENT_PROTOCOL v0.2.

Density/fraction tokens are integer percentages (5..90 / 90,95,100).
SeedRow is immutable; export JSON objects with row._asdict(). A scenario_key
identifies density/run, deliberately shared across fractions and policies.

Evaluator rows are flat mappings. REQUIRED_FIELDS documents the minimal analysis
contract; additional physical metrics/counters/backend metadata may be retained.
training_seed=None means a fixed control, never a synthetic training replicate.
joint_checkpoint_id identifies the frozen snapshot (or fixed controller config).
config_hash covers the common simulator/geometry/measurement configuration, not
policy architecture. protocol_hash covers the frozen experiment definition.
role_tags_validated=True attests creation-time, per-mode tag validation: B1 has
native non-PV tags, other modes have the planned tagged subset, and PV tags are
unchanged. Hash-only analysis cannot establish that attestation itself.
"""

from collections import defaultdict
import hashlib
import json
from typing import NamedTuple

import numpy as np

FAMILIES = ("A", "P_L", "P_G", "J_L", "J_G")
FIXED_CONTROLS = ("B1", "B2", "N", "L", "G_old", "M_old", "C_old")
DIAGNOSTICS = tuple(f"D{i}_{a}" for a in ("L", "G") for i in range(1, 5))
TRAINING_SEEDS = (20260914, 20260915, 20260916)
DENSITY_TOKENS = tuple(range(5, 91, 5))
FRACTION_TOKENS = (90, 95, 100)
TRAINING_DENSITY_IDS = (1, 3, 5, 7)
DIAGNOSTIC_DENSITY_IDS = (3, 5, 7, 11)
VALIDATION_ROOT, TEST_ROOT, DEVELOPMENT_ROOT, BOOTSTRAP_ROOT = 610003, 710009, 810017, 910019
SEED_SCHEMA = "sep_canonical_v1"
SEED_FIELDS = ("scenario_seed", "warmup_rng_seed", "measurement_rng_seed",
               "assignment_seed", "av_selection_seed")
SEED_OFFSETS = (11, 97, 193, 1000, 2000)
METRICS = ("mean_speed_priority", "flow_background", "stopped_fraction_priority",
           "mean_priority_lap_time")
PHYSICAL_FIELDS = ("vehicle_id", "lane", "pos", "vel", "length", "alive",
                   "last_lane_delta", "changed_lane", "controlled")
ROLE_FIELDS = ("veh_type", "behavior_id")
HASH_FIELDS = ("warmed_state_hash", "pre_av_tag_state_hash", "physical_state_hash",
               "pv_role_hash", "role_state_hash", "activated_state_hash", "av_subset_hash")
KEY_FIELDS = ("protocol_hash", "split", "method", "training_seed", "joint_checkpoint_id",
              "canonical_density_id", "fraction_token", "run_index")
REQUIRED_FIELDS = tuple(dict.fromkeys(KEY_FIELDS + SEED_FIELDS + HASH_FIELDS + METRICS + (
    "seed_schema", "split_root", "density_token", "scenario_key", "config_hash",
    "measure_steps", "priority_vehicle_id", "role_tags_validated", "status", "reuse_key")))


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def canonical_density_id(token):
    """Return the full-grid index from an integer percentage, never float equality."""
    _integer(token, "density_token", 1)
    if token not in DENSITY_TOKENS:
        raise ValueError("density_token must be 5,10,...,90")
    return token // 5 - 1


class SeedRow(NamedTuple):
    seed_schema: str
    split: str
    split_root: int
    canonical_density_id: int
    density_token: int
    fraction_token: int
    run_index: int
    scenario_key: str
    scenario_seed: int
    warmup_rng_seed: int
    measurement_rng_seed: int
    assignment_seed: int
    av_selection_seed: int


def seed_manifest(split, root, *, density_tokens=DENSITY_TOKENS,
                  fraction_tokens=FRACTION_TOKENS, runs=30):
    """Expand a sorted immutable manifest. Subsets retain full-grid density IDs."""
    _integer(root, "root")
    _integer(runs, "runs", 1)
    if not isinstance(split, str) or not split:
        raise ValueError("split must be nonempty")
    densities, fractions = tuple(density_tokens), tuple(fraction_tokens)
    if not densities or len(set(densities)) != len(densities):
        raise ValueError("empty/duplicate densities")
    for token in densities:
        canonical_density_id(token)
    if (not fractions or len(set(fractions)) != len(fractions)
            or any(type(f) is not int or f not in FRACTION_TOKENS for f in fractions)):
        raise ValueError("invalid/duplicate fraction tokens")
    rows = []
    for token in sorted(densities):
        index = canonical_density_id(token)
        for fraction in sorted(fractions):
            for run in range(runs):
                base = root + index * 1_000_000 + run * 10_000
                rows.append(SeedRow(SEED_SCHEMA, split, root, index, token, fraction, run,
                                    f"{SEED_SCHEMA}:{root}:{index}:{run}",
                                    *(base + offset for offset in SEED_OFFSETS)))
    validate_seed_manifest(rows)
    return tuple(rows)


def validate_seed_manifest(rows):
    """Validate actual expanded tuples; allow reuse only across fractions."""
    if not rows:
        raise ValueError("empty seed manifest")
    keys, scenarios = set(), {}
    for row in rows:
        if not isinstance(row, SeedRow):
            raise ValueError("manifest entries must be SeedRow objects")
        _integer(row.split_root, "split_root")
        _integer(row.run_index, "run_index")
        _integer(row.canonical_density_id, "canonical_density_id")
        for field in SEED_FIELDS:
            _integer(getattr(row, field), field)
        if (row.seed_schema != SEED_SCHEMA or not row.split
                or canonical_density_id(row.density_token) != row.canonical_density_id
                or type(row.fraction_token) is not int or row.fraction_token not in FRACTION_TOKENS):
            raise ValueError("invalid seed schema/condition")
        base = row.split_root + row.canonical_density_id * 1_000_000 + row.run_index * 10_000
        expected = tuple(base + offset for offset in SEED_OFFSETS)
        if tuple(getattr(row, f) for f in SEED_FIELDS) != expected:
            raise ValueError("expanded seeds do not match canonical derivation")
        if row.scenario_key != f"{SEED_SCHEMA}:{row.split_root}:{row.canonical_density_id}:{row.run_index}":
            raise ValueError("invalid scenario_key")
        key = (row.split, row.canonical_density_id, row.fraction_token, row.run_index)
        if key in keys:
            raise ValueError("duplicate manifest row")
        keys.add(key)
        owner = (row.split, row.split_root, row.canonical_density_id, row.run_index)
        previous = scenarios.setdefault(row.scenario_seed, owner)
        if previous != owner:
            raise ValueError("expanded scenario collision")
    if len({(r.split, r.split_root) for r in rows}) != 1:
        raise ValueError("mixed manifest splits/roots")


def validate_split_separation(*manifests):
    """Reject cross-split tuple, scenario, or same-role RNG seed intersections.

    Actual training draws may be supplied as SeedRows only if they use this
    derivation. For arbitrary training draws use validate_reserved_seeds instead.
    """
    for rows in manifests:
        validate_seed_manifest(rows)
    validate_reserved_seeds({rows[0].split: [tuple(getattr(r, f) for f in SEED_FIELDS)
                                            for r in rows] for rows in manifests})
    if len({rows[0].split for rows in manifests}) != len(manifests):
        raise ValueError("split names must be distinct")


def validate_reserved_seeds(split_tuples):
    """Check actual five-seed tuples by split, including arbitrary training draws.

    Order is SEED_FIELDS. Repeated tuples within a split are permitted (fraction
    or cache reuse). Call before accepting a training draw; deterministic redraw
    and its logging belong to the training caller, not this pure validator.
    """
    owners = [{} for _ in SEED_FIELDS]
    for split, tuples in split_tuples.items():
        for seeds in tuples:
            if len(seeds) != 5:
                raise ValueError("expected five seeds")
            for field, value, seen in zip(SEED_FIELDS, seeds, owners):
                _integer(value, field)
                if value in seen and seen[value] != split:
                    raise ValueError(f"split collision: {field}={value} in {seen[value]} and {split}")
                seen[value] = split


def _array_hash(arrays):
    # JSON records fix field names, integer values, shape and order, without repr,
    # platform byte order or NumPy dtype-width dependence.
    records = []
    for name, values in arrays:
        arr = np.asarray(values)
        if arr.ndim != 1 or arr.dtype.kind not in "biu":
            raise ValueError(f"{name} must be a one-dimensional integer/bool array")
        records.append([name, [int(v) for v in arr]])
    return hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()


def state_fingerprints(arrays, *, priority_vehicle_id, av_vehicle_ids):
    """Hash explicit TrafficState-field mappings; rows are sorted by vehicle ID.

    Call on warm, PV-assigned pre-tag, and activated snapshots. Use the returned
    activated_state_hash as warmed_state_hash/pre_av_tag_state_hash for the first
    two calls. ordered AV IDs retain selection order; their hash is not sorted.
    This function fingerprints state, it does not validate native physics/tags.
    """
    _integer(priority_vehicle_id, "priority_vehicle_id")
    fields = PHYSICAL_FIELDS + ROLE_FIELDS
    missing = set(fields) - arrays.keys()
    if missing:
        raise ValueError(f"missing state arrays: {sorted(missing)}")
    data = {f: np.asarray(arrays[f]) for f in fields}
    n = data["vehicle_id"].size
    if any(a.shape != (n,) or a.dtype.kind not in "biu" for a in data.values()):
        raise ValueError("state fields must be equally sized 1D integer/bool arrays")
    ids = data["vehicle_id"]
    if len(np.unique(ids)) != n or priority_vehicle_id not in ids:
        raise ValueError("duplicate IDs or absent priority_vehicle_id")
    av = np.asarray(av_vehicle_ids)
    if av.size == 0:
        av = np.array([], dtype=np.int64)
    if (av.ndim != 1 or av.dtype.kind not in "iu" or len(np.unique(av)) != av.size
            or priority_vehicle_id in av or not np.all(np.isin(av, ids))):
        raise ValueError("invalid ordered AV subset")
    order = np.argsort(ids, kind="stable")
    ordered = {f: a[order] for f, a in data.items()}
    pv = ordered["vehicle_id"] == priority_vehicle_id
    return {
        "physical_state_hash": _array_hash((f, ordered[f]) for f in PHYSICAL_FIELDS),
        "role_state_hash": _array_hash((f, ordered[f]) for f in ("vehicle_id",) + ROLE_FIELDS),
        "pv_role_hash": _array_hash((f, ordered[f][pv]) for f in ("vehicle_id",) + ROLE_FIELDS),
        "activated_state_hash": _array_hash((f, ordered[f]) for f in fields),
        "av_subset_hash": _array_hash((("av_vehicle_ids", av),)),
    }


def fixed_reuse_key(method, seed_row):
    """Physical trajectory identity; B1 ignores fraction, all fixed modes ignore training seed."""
    if method not in FIXED_CONTROLS:
        raise ValueError("reuse keys apply only to fixed controls")
    suffix = "" if method == "B1" else f":f{seed_row.fraction_token}"
    return f"{method}:{seed_row.scenario_key}{suffix}"


def validate_pair(left, right):
    """Check provenance before joining metrics. Rows must first pass validate_rows."""
    common = ("protocol_hash", "split", "config_hash", "seed_schema", "split_root",
              "canonical_density_id", "density_token", "fraction_token", "run_index",
              "scenario_key", "measure_steps", "priority_vehicle_id", "warmed_state_hash",
              "pre_av_tag_state_hash", "physical_state_hash", "pv_role_hash") + SEED_FIELDS
    if left["method"] != "B1" and right["method"] != "B1":
        common += ("role_state_hash", "activated_state_hash", "av_subset_hash")
    for field in common:
        if left[field] != right[field]:
            raise ValueError(f"pair mismatch: {field}")
    if left["role_tags_validated"] is not True or right["role_tags_validated"] is not True:
        raise ValueError("creation-time role tags must be validated")


def validate_rows(rows, *, manifest, checkpoints):
    """Validate a complete rectangular matrix before any dict join; return None.

    checkpoints: {(method, training_seed_or_None): joint_checkpoint_id} defines
    exact expected instances. Supply all 22 for full test, or an explicit subset
    for a focused comparison. manifest defines exact conditions/runs. Failed rows
    are rejected for analysis, retained by evaluator in its failure artifact.
    Extra evaluator fields are allowed, but only this module's fields are checked.
    Development-only tiny/pilot runs may use other nonnegative integer training
    seeds; validation/test retain the three preregistered training seeds.
    """
    validate_seed_manifest(manifest)
    if not checkpoints:
        raise ValueError("empty checkpoint map")
    learned_ids = []
    for (method, seed), checkpoint in checkpoints.items():
        if method in FIXED_CONTROLS:
            if seed is not None:
                raise ValueError("fixed controls must have training_seed=None")
        elif method in FAMILIES + DIAGNOSTICS:
            if (type(seed) is not int or seed < 0
                    or (manifest[0].split != "development" and seed not in TRAINING_SEEDS)):
                raise ValueError("invalid training seed")
            if method in FAMILIES:
                learned_ids.append(checkpoint)
        else:
            raise ValueError(f"unknown method: {method}")
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError("empty checkpoint ID")
    if len(set(learned_ids)) != len(learned_ids):
        raise ValueError("duplicate learned checkpoint IDs")
    planned = {(r.canonical_density_id, r.fraction_token, r.run_index): r for r in manifest}
    expected = {(method, seed, *condition) for method, seed in checkpoints for condition in planned}
    seen, configs, groups, reused = set(), set(), defaultdict(list), {}
    for row in rows:
        missing = set(REQUIRED_FIELDS) - row.keys()
        if missing:
            raise ValueError(f"missing row fields: {sorted(missing)}")
        for field in ("canonical_density_id", "density_token", "fraction_token", "run_index",
                      "split_root", "priority_vehicle_id", "measure_steps") + SEED_FIELDS:
            _integer(row[field], field)
        if row["measure_steps"] < 1:
            raise ValueError("measure_steps must be positive")
        if row["training_seed"] is not None:
            _integer(row["training_seed"], "training_seed")
        condition = tuple(row[f] for f in ("canonical_density_id", "fraction_token", "run_index"))
        key = (row["method"], row["training_seed"], *condition)
        if key in seen:
            raise ValueError("duplicate run row")
        if key not in expected:
            raise ValueError("unexpected row instance/condition")
        seen.add(key)
        if row["joint_checkpoint_id"] != checkpoints[key[:2]]:
            raise ValueError("checkpoint mismatch")
        for field, value in planned[condition]._asdict().items():
            if row[field] != value:
                raise ValueError(f"manifest mismatch: {field}")
        for field in ("protocol_hash", "config_hash") + HASH_FIELDS:
            value = row[field]
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"{field} must be a lowercase SHA256 hex digest")
        configs.add((row["protocol_hash"], row["config_hash"], row["measure_steps"]))
        if row["status"] != "ok" or row["role_tags_validated"] is not True:
            raise ValueError("row failed or creation-time role tags unvalidated")
        for field in METRICS:
            value = row[field]
            if value is None and field == "mean_priority_lap_time":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"{field} must be finite (only lap time may be null)")
            if value < 0 or (field == "stopped_fraction_priority" and value > 1):
                raise ValueError(f"invalid {field}")
        if row["method"] in FIXED_CONTROLS:
            reuse = fixed_reuse_key(row["method"], planned[condition])
            if row["reuse_key"] != reuse:
                raise ValueError("fixed-control reuse identity mismatch")
            signature = tuple(row[f] for f in METRICS + HASH_FIELDS + ("priority_vehicle_id",))
            if reuse in reused and reused[reuse] != signature:
                raise ValueError("reused fixed trajectory output mismatch")
            reused[reuse] = signature
        elif row["reuse_key"] is not None:
            raise ValueError("learned rows must have reuse_key=None")
        groups[condition].append(row)
    if seen != expected:
        raise ValueError(f"missing rows: {len(expected - seen)}")
    if len(configs) != 1:
        raise ValueError("mixed protocol/config/horizon")
    scenarios = {}
    for condition, group in groups.items():
        # Compare all non-B1 to a non-B1 anchor, not only each to B1.
        anchor = next((r for r in group if r["method"] != "B1"), group[0])
        for row in group:
            validate_pair(anchor, row)
            scenario = (condition[0], condition[2])
            signature = tuple(row[f] for f in SEED_FIELDS + (
                "warmed_state_hash", "pre_av_tag_state_hash", "physical_state_hash", "pv_role_hash",
                "priority_vehicle_id"))
            if scenario in scenarios and scenarios[scenario] != signature:
                raise ValueError("fraction block provenance mismatch")
            scenarios[scenario] = signature


def planned_counts():
    """Nominal arithmetic only, not completed work or measured runtime."""
    conditions, runs, seeds = 18 * 3, 30, len(TRAINING_SEEDS)
    rows = conditions * runs * (len(FIXED_CONTROLS) + len(FAMILIES) * seeds)
    trajectories = 18 * runs + conditions * runs * (6 + 5 * seeds)
    diagnostics = 12 * runs * seeds * 8
    validation = 5 * seeds * 5 * 20
    return dict(test_logical_rows=rows, test_trajectories=trajectories,
                test_measurement_steps=trajectories * 2000, test_shared_warmup_steps=18 * runs * 1000,
                diagnostic_trajectories=diagnostics, diagnostic_measurement_steps=diagnostics * 2000,
                validation_policy_trajectories=validation, validation_policy_measurement_steps=validation * 2000,
                training_runs=5 * seeds, training_env_steps_nominal=5 * seeds * 20_000_000)
