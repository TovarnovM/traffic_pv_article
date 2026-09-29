"""4.5 step 4 tie-break executor: 50 fresh-seed runs for the top two checkpoints.

The frozen validation gate (evaluate.prepare_plan, purpose='validation') requires
exactly the preregistered 20-run root-610003 manifest, so the 50-run tie-break is
executed by this labelled variant using evaluate's own simulation primitives
(prepare_group, measure, load_pair). No frozen module is modified. Tie-break
tasks are derived only from an existing pair decision.
"""
from pathlib import Path

from sep_research import evaluate, protocol
from sep_research.export_legacy import json_hash
from sep_research.eval_selection_10m import selection as sel

SCHEMA_TIEBREAK = 'sep_tiebreak_10m_v1'
_MODELS, _PHYSICAL = {}, None


def _namespace_is_clean(rows):
    """Reject a tie-break whose five-seed tuples intersect any reserved split seed."""
    from sep_research import protocol as pr
    from sep_research.seed_control import frozen_seed_guard
    blocked = set(frozen_seed_guard()['reserved_values'])
    for root, density, fractions, runs in (
            (pr.VALIDATION_ROOT, (30,), (100,), 20),
            (pr.TEST_ROOT, pr.DENSITY_TOKENS, pr.FRACTION_TOKENS, 30),
            (pr.DEVELOPMENT_ROOT, pr.DENSITY_TOKENS, pr.FRACTION_TOKENS, 30),
            (pr.BOOTSTRAP_ROOT, pr.DENSITY_TOKENS, pr.FRACTION_TOKENS, 30)):
        for row in pr.seed_manifest('validation', root, density_tokens=density,
                                    fraction_tokens=fractions, runs=runs):
            blocked |= {getattr(row, field) for field in pr.SEED_FIELDS}
    values = {getattr(row, field) for row in rows for field in pr.SEED_FIELDS}
    if values & blocked:
        raise ValueError('tie-break seed namespace collides with reserved split seeds')
    return True


def tiebreak_manifest(*, runs=sel.TIEBREAK_RUNS, root=sel.TIEBREAK_ROOT):
    rows = protocol.seed_manifest('validation', root, density_tokens=sel.VALIDATION_DENSITIES,
                                  fraction_tokens=sel.VALIDATION_FRACTIONS, runs=runs)
    _namespace_is_clean(rows)
    return rows


def tiebreak_tasks(origins, tie):
    """One task per fresh-seed run; each task carries its B2 control and both candidates."""
    by_pair = {(r['family'], r['seed']): r for r in origins}
    origin = by_pair[(tie['family'], tie['seed'])]
    bundles = {iteration: next(c for c in origin['candidates'] if c['iteration'] == iteration)
               for iteration in tie['iterations']}
    tasks = []
    for row in tiebreak_manifest():
        tasks.append(dict(schema=SCHEMA_TIEBREAK, family=tie['family'], seed=tie['seed'],
                          run_index=row.run_index, seed_row=row._asdict(),
                          candidates=[dict(iteration=iteration, bundle=bundles[iteration]['path'],
                                           run_config=bundles[iteration]['run_config'],
                                           manifest_sha256=bundles[iteration]['manifest_sha256'])
                                      for iteration in tie['iterations']]))
    if len(tasks) != sel.TIEBREAK_RUNS or len({task['run_index'] for task in tasks}) != sel.TIEBREAK_RUNS:
        raise ValueError('tie-break must contain exactly one task per fresh-seed run')
    return tasks


def _init(tasks):
    global _MODELS, _PHYSICAL
    import torch
    torch.set_num_threads(1)
    _PHYSICAL = dict(evaluate.PHYSICAL_DEFAULTS)
    task = tasks[0]
    _MODELS = {}
    for candidate in task['candidates']:
        contract = evaluate._read_json(Path(candidate['run_config']))['contract']
        models, manifest = evaluate.load_pair(Path(candidate['bundle']), contract)
        if json_hash(manifest['contract']) != json_hash(contract):
            raise ValueError('tie-break bundle changed')
        _MODELS[(task['family'], candidate['iteration'])] = models


def _run_task(task):
    from sep_research.protocol import SeedRow
    seed_row = SeedRow(**task['seed_row'])
    _, pre, pv, states, _ = evaluate.prepare_group(_PHYSICAL, [seed_row])
    initial, av = states[seed_row.fraction_token]
    evaluate._MODELS.clear()
    evaluate._MODELS.update(_MODELS)
    baseline = evaluate.measure({'plan': {'physical': _PHYSICAL}}, seed_row, 'B2', None, initial, pv, av)
    per_run = []
    for candidate in task['candidates']:
        evaluate._MODELS.clear()
        evaluate._MODELS.update({(task['family'], task['seed']):
                                 _MODELS[(task['family'], candidate['iteration'])]})
        result = evaluate.measure({'plan': {'physical': _PHYSICAL}}, seed_row, task['family'],
                                  task['seed'], initial, pv, av)
        per_run.append((candidate['iteration'], result['mean_speed_priority']))
    return per_run, baseline['mean_speed_priority']


def execute_tiebreak(tasks, *, workers=1):
    """Run the tie-break and return selection.tiebreak_decision input plus B2 speeds."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    iterations = [candidate['iteration'] for candidate in tasks[0]['candidates']]
    speeds = {iteration: [] for iteration in iterations}
    baseline = []
    if workers == 1:
        _init(tasks)
        results = [_run_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_init, initargs=(tasks,)) as pool:
            results = list(pool.map(_run_task, tasks))
    for per_run, b2 in results:
        for iteration, speed in per_run:
            speeds[iteration].append(speed)
        baseline.append(b2)
    for iteration in iterations:
        if len(speeds[iteration]) != len(tasks):
            raise ValueError('tie-break run coverage mismatch')
    return dict(run_speeds={str(iteration): speeds[iteration] for iteration in iterations},
                b2_speeds=baseline, runs=len(tasks))
