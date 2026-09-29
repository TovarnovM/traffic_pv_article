"""Explicit operational authorization, never measured pilot acceptance.

The release and adjacent summaries are trusted operator inputs, with integrity
checks rather than authentication, like the existing acceptance manifests.
"""
from .common import ROOT, digest, read, relative, safe, sha
from .contract import validate_plan

DEFAULT_PATH = ROOT / 'sep_research/cluster_cpu/operator_release.json'
SCHEMA = 'cpu_operator_release_v1'
MODE = 'operator_acceptance_without_additional_pilots'
PLAN_HASH = '84e89193d8ff3187a4601337abc2f995d4883f7c245187517a31e40a7ac992d3'
PROFILE_HASH = '1a584afeec6b7d35a86845099e00a87915e3864cf9bd306034dcf99f7ec78317'
DECISION = ('Считаю, что все проверки выполнены - больше ничего не надо проверять. '
            'Дальше - уже боевой запуск.')
SCOPE = 'fresh15x5M_only; extensions_to_6M_or_10M_require_new_human_request_and_release'
LIMITATIONS = {
    'ten_iteration_pilots': 'Waived: no 10+ iteration family pilots, warm-cache/condition coverage or pilot restore evidence; only one smoke iteration per family and one P_L continuation.',
    'hardware_36_logical_cpus': 'Waived: smoke hosts had 72 logical CPUs; Docker quota36 is not evidence on a 36-logical-CPU host.',
    'sustained_spill_and_memory': 'Waived: graph smoke spilled/restored objects; sustained throughput, absence of thrashing and long-run memory reserve are unmeasured, including concurrent graph pilots.',
}
EVIDENCE = {
    'remote_evidence/smoke_summary.json': '853271665a05b3b8edc216a0bd7adbce0beb7953b9a83b870bf480e343aa2fa5',
    'remote_evidence/resume_summary.json': '47a27f7d653cf6e001939c432ebbc9a116fe855df21a630904dd89dcfbe04eca',
}



def production_release(plan):
    """Construct the explicit fresh-v3 decision; never authorize revisions."""
    from .contract import plan as make_plan, production_profile
    validate_plan(plan)
    if digest(plan) != digest(make_plan(profile=production_profile())):
        raise ValueError('operator release does not authorize this exact fresh 15x5M plan')
    return dict(schema=SCHEMA, mode=MODE, plan_hash=digest(plan),
                profile_hash=plan['profile_hash'], scope=SCOPE,
                human_decision='Ок, запускай 16 worker config в дело! Уже нужен боевой запуск',
                operational_authorization='authorized', measured_acceptance=False,
                limitations=dict(LIMITATIONS,
                    graph_8gib_store='Explicitly authorized without full performance acceptance: graph 8GiB object-store testing was stopped; sustained memory, spill and throughput remain unverified.'),
                evidence={})


def production_continuation_release(plan):
    """Authorize only the requested completed-v3 5M -> 10M learner continuation.

    Source IDs and result hashes are bound by the plan digest, not substitutes for
    revision/_source/validate_result's completed-source and scientific checks.
    No interrupted prefix or recovery route is authorized here.
    """
    from .contract import plan as make_plan, production_profile
    validate_plan(plan)
    previous = make_plan(profile=production_profile())
    # Canonical digest of the immutable production_v3/plan.json, not a live file lookup.
    previous_hash = '64d6e655ae044994f2e313cf42ed2063adb3167773baaf46fd0ffe109fe3ef59'
    expected = make_plan(budget=10_000_000, profile=production_profile())
    expected['previous_plan_hash'] = previous_hash
    for row, source in zip(expected['runs'], plan['runs']):
        row.update(start_iteration=306, segment_boundaries=[0, 306, 611],
                   source_result='inputs/' + row['run_id'],
                   source_checkpoint_id=source['source_checkpoint_id'],
                   source_result_hash=source['source_result_hash'])
    if digest(previous) != previous_hash or digest(plan) != digest(expected):
        raise ValueError('operator release does not authorize this exact 15x10M continuation plan')
    record = production_release(previous)
    record.update(plan_hash=digest(plan), previous_plan_hash=previous_hash,
                  scope='completed_production_v3_15x5M_to_10M_only; boundaries_0_306_611; no_interrupted_prefix_recovery',
                  human_decision='Вариант 1 — продолжить текущие до 10 млн - ДА! Вперед')
    record['limitations'].update(
        segmented_learner_restore='Only completed terminal 5M sources: learner weights, critics, Adam slots, KL coefficients, gradient updates, counters and iteration are restored and verified; interrupted prefixes are not accepted.',
        continuation_rng_reset='Segmented learner continuation, not exact uninterrupted resume: environments, cache, episodes and Python/NumPy/Torch sampling RNG restart deterministically at iteration 306 under sep_amendment_segment_rng_v1; original training seeds and seed-exclusion checks remain unchanged.')
    return record


def production_recovery_release(plan):
    """Authorize the explicitly chosen J_G150 recovery and its 10M continuation."""
    from .contract import plan as make_plan, production_profile
    validate_plan(plan)
    if (plan['schema'] != 'cpu_plan_v2'
            or plan.get('purpose') not in ('interrupted_recovery_5m', 'recovered_matrix_10m')):
        raise ValueError('operator release requires the exact staged recovery plan')
    previous = make_plan(profile=production_profile())
    if plan['previous_plan_hash'] != digest(previous):
        raise ValueError('recovery campaign anchor differs from production v3')
    record = production_release(previous)
    record.update(plan_hash=digest(plan), previous_plan_hash=digest(previous),
                  scope=plan['purpose'],
                  human_decision='Вариант 1 — продолжить текущие до 10 млн - ДА! Вперед',
                  recovery_decision='Восстановить с checkpoint 150')
    record['limitations'].update(
        interrupted_prefix='Only J_G seed20260916 checkpoint150: original failed attempt is preserved; no full prefix lifecycle/seed census is available. Native learner and checkpoint metric fingerprints must match. Old iterations151-158 and partial work are discarded, not credited.',
        recovery_eligibility='J_G recovery ancestry remains prefix_audit_complete=false and original_protocol_eligible=false, including after the final611 endpoint. A completed recovery suffix never certifies the failed original prefix.',
        staged_rng_reset='Segmented learner restoration preserves weights, critics, Adam, KL, gradient updates and counters; environments/cache/episodes/sampling RNG restart deterministically at150 and306 for the recovered J_G, at306 for ordinary continuations. Historical segment boundaries differ; not uninterrupted or uniformly matched-history training.')
    return record


def production_cohort_release(plan):
    """Authorize only this ready subset; later completed lines need their own release."""
    from .contract import plan as make_plan, production_profile
    validate_plan(plan)
    if plan['schema'] != 'cpu_plan_v2' or plan.get('purpose') != 'ready_cohort_10m':
        raise ValueError('operator release requires the exact ready cohort plan')
    previous = make_plan(profile=production_profile())
    record = production_release(previous)
    record.update(plan_hash=digest(plan), previous_plan_hash=digest(previous),
                  scope='ready_cohort_10m',
                  human_decision='А можно как-то пока запустить то, что уже обучилось до 306? А оставшиеся 5 запустить чуть позже (когда они досчитаются)',
                  continuation_decision='Вариант 1 — продолжить текущие до 10 млн - ДА! Вперед',
                  recovery_decision='Восстановить с checkpoint 150')
    record['limitations'].update(
        ready_sources_only='Only the plan-bound nonempty subset of original production v3 identities with validated completed306 sources; total budget10M, endpoint611. No pending rows or authorization of later cohorts.',
        recovery_eligibility='Only J_G seed20260916 may use the completed150→306 recovery; its boundaries remain0,150,306,611 and prefix_audit_complete=false and original_protocol_eligible=false. Other lines require original completed306 sources and boundaries0,306,611.',
        segmented_learner_restore='Weights, critics, Adam, KL, gradient updates and counters are restored and verified; environments/cache/episodes/sampling RNG restart deterministically at306 (also at150 in J_G recovery ancestry). Not exact uninterrupted resume; a completed recovery suffix never certifies the failed original prefix.')
    return record


def production_partial_continuation_release(plan):
    """Authorize only the single killed-attempt P_G14 resume from checkpoint 575."""
    from .contract import plan as make_plan, production_profile
    from .partial import PREVIOUS_PLAN_HASH
    validate_plan(plan)
    if plan['schema'] != 'cpu_plan_v2' or plan.get('purpose') != 'interrupted_continuation_10m':
        raise ValueError('operator release requires the exact partial continuation plan')
    previous = make_plan(profile=production_profile())
    if plan['previous_plan_hash'] != PREVIOUS_PLAN_HASH:
        raise ValueError('partial continuation anchor differs from the ready cohort plan')
    record = production_release(previous)
    record.update(plan_hash=digest(plan), previous_plan_hash=plan['previous_plan_hash'],
                  scope='interrupted_continuation_10m',
                  human_decision='Подними P_G14 на pc-411w',
                  resume_decision='Восстановить P_G14 с checkpoint 575 (iterations 576..586 discarded)')
    record['limitations'].update(
        partial_prefix=('Only P_G14 attempt_002 checkpoint 575: the killed attempt computed through 586 '
                        'but never checkpointed iterations 576..586, so those iterations are discarded and '
                        'never credited. Weights, critics, Adam slots, KL coefficients, gradient updates and '
                        'counters are verified only at the saved 575 bundle.'),
        resume_boundary=('The ready-cohort segment 306→611 is split by an extra authorized resume boundary '
                         'at 575, so the line history is heterogeneous and boundary_comparability stays '
                         'heterogeneous_authorized_resume. Earlier campaign candidates 450/500/550 remain '
                         'owned by the preserved attempt_002 source and are not re-declared by this plan.'),
        segmented_rng_reset=('Segmented learner continuation, not exact uninterrupted resume: environments, '
                             'cache, episodes and Python/NumPy/Torch sampling RNG restart deterministically at '
                             '575 under sep_amendment_segment_rng_v1; original training seeds, profile, resource '
                             'geometry and seed-exclusion checks are unchanged.'))
    return record


def production_recovery_resume_release(plan):
    """Authorize only the single killed-recovery J_G16 resume from checkpoint 275."""
    from .contract import plan as make_plan, production_profile
    from .recovery_partial import PREVIOUS_PLAN_HASH
    validate_plan(plan)
    if plan['schema'] != 'cpu_plan_v2' or plan.get('purpose') != 'interrupted_recovery_resume_5m':
        raise ValueError('operator release requires the exact recovery resume plan')
    previous = make_plan(profile=production_profile())
    if plan['previous_plan_hash'] != PREVIOUS_PLAN_HASH:
        raise ValueError('recovery resume anchor differs from the interrupted recovery plan')
    record = production_release(previous)
    record.update(plan_hash=digest(plan), previous_plan_hash=plan['previous_plan_hash'],
                  scope='interrupted_recovery_resume_5m',
                  human_decision='Запускай',
                  resume_decision='Восстановить J_G16 с checkpoint 275 (iterations 276..297 discarded)')
    record['limitations'].update(
        partial_prefix=('Only J_G16 attempt_002 checkpoint 275: the killed recovery attempt computed '
                        'through 297 but never checkpointed iterations 276..297, so those iterations are '
                        'discarded and never credited. Weights, critics, Adam slots, KL coefficients, '
                        'gradient updates and counters are verified only at the saved 275 bundle.'),
        recovery_eligibility=('J_G recovery ancestry stays prefix_audit_complete=false and '
                              'original_protocol_eligible=false, including after the final306 endpoint. '
                              'A completed resume suffix never certifies the failed original prefix or the '
                              'discarded 151..158 and 276..297 iterations.'),
        resume_boundary=('The 150→306 recovery is split by an extra authorized resume boundary at 275, so '
                         'the line history is heterogeneous and boundary_comparability stays '
                         'heterogeneous_authorized_resume. The earlier 150→306 attempt_002 source and the '
                         'failed attempt_001 prefix remain preserved, not rewritten.'),
        segmented_rng_reset=('Segmented learner continuation, not exact uninterrupted or bitwise resume: '
                             'environments, cache, episodes and Python/NumPy/Torch sampling RNG restart '
                             'deterministically at the 275 boundary under sep_amendment_segment_rng_v1; '
                             'original training seeds, profile, resource geometry and seed-exclusion checks '
                             'are unchanged.'))
    return record


def verify_record(value, plan):
    """Validate the native authorization record independently of its location."""
    validate_plan(plan)
    if plan['schema'] == 'cpu_plan_v2':
        if plan.get('purpose') == 'ready_cohort_10m':
            expected = production_cohort_release(plan)
        elif plan.get('purpose') == 'interrupted_recovery_resume_5m':
            expected = production_recovery_resume_release(plan)
        elif plan.get('purpose') == 'interrupted_continuation_10m':
            expected = production_partial_continuation_release(plan)
        else:
            expected = production_recovery_release(plan)
    elif plan['profile']['schema'] == 'cpu_profile_v3':
        expected = (production_continuation_release(plan) if plan['previous_plan_hash']
                    else production_release(plan))
    else:
        if digest(plan) != PLAN_HASH or plan['profile_hash'] != PROFILE_HASH:
            raise ValueError('operator release does not authorize this exact fresh 15x5M plan')
        expected = dict(schema=SCHEMA, mode=MODE, plan_hash=PLAN_HASH,
                        profile_hash=PROFILE_HASH, scope=SCOPE, human_decision=DECISION,
                        operational_authorization='authorized', measured_acceptance=False,
                        limitations=LIMITATIONS, evidence=EVIDENCE)
    if digest(value) != digest(expected):
        raise ValueError('invalid explicit operator release record')
    return value


def verify(path, plan):
    """Verify this narrowly scoped release and its portable summary bytes."""
    path = safe(path)
    value = verify_record(read(path), plan)
    for name, expected_hash in value['evidence'].items():
        if sha(relative(path.parent, name)) != expected_hash:
            raise ValueError('operator release smoke summary hash mismatch')
    return value
