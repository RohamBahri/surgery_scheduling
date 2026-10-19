"""Fast, read-only training audit. No new MIP solves, no test outcomes."""
import hashlib
import json
from collections import Counter

import numpy as np

from pilot import run_final_2013 as run
from surgery.anchored_core import key
from surgery.planner import display_and_duration


def cached_deployment(folder, X, booked, weights):
    """Read exact training certificates already computed by ANCHORED."""
    _, duration = display_and_duration(X @ weights, booked, run.ALPHA, run.H)
    digest = hashlib.sha256(np.ascontiguousarray(duration, dtype=np.float64).tobytes()).hexdigest()
    path = folder / f'anchored_verify_{digest}.json'
    return json.loads(path.read_text())['components'] if path.is_file() else {}


def audit_training(args):
    run.settings(args)
    data, weeks, _ = run.training(args, 'cohort')
    home = args.root / 'cohort'
    report = {'training_cases': len(data['booked']),
              'site_weeks': len(weeks),
              'new_solver_calls': 0,
              'solver_diagnostics': {},
              'operational_diagnostics': {},
              'notes': ['Diagnostic, not an operational feasibility certificate.',
                        'No 2013 test outcomes examined.']}
    paths = run.available_methods(args.root)
    for name, path in paths.items():
        obj = json.loads(path.read_text())
        if name.startswith('CASE'):
            bic = json.loads((path.parent / 'case_bic_path.json').read_text())
            selected = min(bic, key=lambda a: (a['bic_proxy'], a['lambda_l1']))
            report['solver_diagnostics'][name] = {
                'lambda': selected['lambda_l1'],
                'optimizer_passes': selected['optimization']}
        elif name.startswith('ANCHORED'):
            master = [record for step in obj.get('history', [])
                      for inner in step.get('inner', [])
                      for record in inner.get('pdca', [])]
            report['solver_diagnostics'][name] = {
                'stopping': obj.get('stopping'),
                'accepted_updates': len(obj.get('accepted', [])),
                'master_solve_count': len(master),
                'incomplete_master_solves': sum(x.get('complete') is False for x in master),
                'separation_closed': obj.get('separation_closed', False),
                'training_complete': obj.get('complete', False)}
        elif name == 'VF_COHORT':
            rows = obj.get('iterations', [])
            report['solver_diagnostics'][name] = {
                'stopping': obj.get('stopping'),
                'outer_rounds': len(rows),
                'accepted_updates': sum(bool(x.get('accepted')) for x in rows),
                'inner_optimizer_not_converged': sum(any(not q.get('optimizer_success', False)
                    for q in x.get('fit_passes', [])) for x in rows)}
        elif name == 'GAP_COHORT':
            history = obj.get('history', [])
            report['solver_diagnostics'][name] = {
                'stopping': obj.get('stopping'),
                'training_complete': obj.get('complete', False),
                'paused': bool(obj.get('paused')),
                'outer_rounds': len(history),
                'last_interval_width': (history[-1].get('certified_after', {})
                    .get('regularized_gap_interval_width_per_case') if history else None)}

    primary = paths.get('ANCHORED_0p001')
    case = paths.get('CASE_COHORT')
    if primary and case:
        root = primary.parent
        weights = lambda p: np.asarray(json.loads(p.read_text())['w'], float)
        a = cached_deployment(root, data['X'], data['booked'], weights(primary))
        b = cached_deployment(root, data['X'], data['booked'], weights(case))
        oracle_path = home / 'train_refs' / 'anchored_references.json'
        refs = json.loads(oracle_path.read_text()) if oracle_path.is_file() else {}
        expected = {key(c) for w in weeks for c in w.components()}
        both = {k for k in expected if a.get(k, {}).get('complete') and b.get(k, {}).get('complete')}
        changes = []
        for k in both:
            aa, bb = a[k], b[k]
            changes.append({'component': k,
                            'cost_difference': aa['realized_cost'] - bb['realized_cost'],
                            'max_load_anchored': aa.get('max_realized_room_load'),
                            'max_load_case': bb.get('max_realized_room_load'),
                            'over_960_anchored': aa.get('room_days_over_960'),
                            'assignment_changed': aa.get('assignment') != bb.get('assignment')})
        differences = sorted(changes, key=lambda x: x['cost_difference'])
        report['operational_diagnostics'] = {
            'certified_cached_pairs': len(both), 'expected_components': len(expected),
            'paired_coverage': len(both) / max(1, len(expected)),
            'training_reference_complete': bool(refs) and all(refs.get(k, {}).get('complete') for k in expected),
            'assignment_changes': sum(x['assignment_changed'] for x in changes),
            'extreme_components_anchored': sum((x['over_960_anchored'] or 0) > 0 for x in changes),
            'top_8_benefits': differences[:8],
            'top_8_harms': differences[-8:][::-1],
            'overload_avoidable': 'not evaluated; requires a separate optional feasibility study'}
        if both:
            total = sum(x['cost_difference'] for x in changes)
            report['operational_diagnostics']['paired_cost_difference_per_training_case'] = total / len(data['booked'])
    out = args.root / 'train_audit' / 'audit_summary.json'
    run.save(out, report)
    print('Fast audit saved:', out, '(no new solver calls)', flush=True)