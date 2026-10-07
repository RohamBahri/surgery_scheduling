"""A resumable experiment for the single UHN workbook."""
import argparse
import hashlib
import logging
import pickle
import sys
import time
from pathlib import Path

import gurobipy as gp
import numpy as np
import pandas as pd

from .data import COLUMNS, GROUPS, SCENARIOS, Encoder, build_cohort, build_days
from .learning import fit_policy, penalty, project, train_shift, train_vf
from .planner import display_and_duration, metrics, model, optimize, response_oracle, solve_day
from .reporting import METHODS, historical_checks, summarize
from .storage import Store, digest, write_json

LOG = logging.getLogger('surgery')


def scenario_key(alpha, h):
    return f'{alpha:g}_{h}'


def prepare(workbook, output, tie, seed):
    source = hashlib.sha256(workbook.read_bytes()).hexdigest()
    code = digest({p.name: p.read_text() for p in sorted(Path(__file__).parent.glob('*.py'))})
    specification = {'input_sha256': source, 'code_sha256': code, 'tie': tie, 'seed': seed,
                     'groups': GROUPS, 'scenarios': SCENARIOS}
    root = output / digest(specification)[:16]
    root.mkdir(parents=True, exist_ok=True)
    write_json(output / 'current_run.json', {'directory': str(root.resolve()), **specification})
    cache = root / 'prepared.pkl'
    if cache.exists():
        with cache.open('rb') as stream:
            payload = pickle.load(stream)
        return root, payload, specification
    LOG.info('Reading workbook and building the audited cohort')
    raw = pd.read_excel(workbook, usecols=COLUMNS)
    cohort, audit, excluded, quality = build_cohort(raw)
    audit['input_sha256'] = source
    audit['input_rows'] = len(raw)
    audit['retained_row_sha256'] = digest(cohort.case_id.to_list())
    audit_dir = root / 'audit'
    audit_dir.mkdir(exist_ok=True)
    cohort[['case_id', 'group', 'date', 'split']].to_csv(audit_dir / 'retained_rows.csv', index=False)
    excluded.to_csv(audit_dir / 'excluded_rows.csv', index=False)
    quality.to_csv(audit_dir / 'daily_cleaning.csv', index=False)
    pd.DataFrame(audit['flow']).to_csv(audit_dir / 'cohort_flow.csv', index=False)
    pd.DataFrame(audit['counts']).to_csv(audit_dir / 'cohort_counts.csv', index=False)
    payload, feasibility = {'groups': {}, 'audit': audit}, []
    for group in GROUPS:
        frames = {split: cohort[cohort.group.eq(group) & cohort.split.eq(split)].reset_index(drop=True)
                  for split in ('train', 'test')}
        if any(frame.empty for frame in frames.values()):
            raise ValueError(f'{group} has an empty training or test set')
        encoder = Encoder()
        X = {'train': encoder.fit_transform(frames['train'])}
        X['test'] = encoder.transform(frames['test'])
        days = {split: build_days(frame, frames['train']) for split, frame in frames.items()}
        payload['groups'][group] = {'frames': frames, 'X': X, 'days': days}
        write_json(audit_dir / f'encoder_{group}.json', encoder.metadata())
        np.savez_compressed(audit_dir / f'features_{group}.npz', X_train=X['train'], X_test=X['test'],
                            training_scores=encoder.training_scores,
                            train_case_ids=frames['train'].case_id.to_numpy(),
                            test_case_ids=frames['test'].case_id.to_numpy())
        for split, bundles in days.items():
            for day in bundles:
                feasibility.append({'group': group, 'split': split, 'date': day.date,
                                    'cases': len(day.booked), 'surgeon_days': day.n_surgeons,
                                    'candidate_rooms': day.n_rooms,
                                    'fallback_surgeons': day.fallback_surgeons,
                                    'fallback_cases': day.fallback_cases,
                                    'assignable': all(bool(e) for e in day.eligible)})
    pd.DataFrame(feasibility).to_csv(audit_dir / 'daily_feasibility.csv', index=False)
    audit['fallback_surgeon_days'] = sum(r['fallback_surgeons'] for r in feasibility)
    audit['all_days_assignable'] = all(r['assignable'] for r in feasibility)
    write_json(audit_dir / 'audit.json', audit)
    write_json(root / 'manifest.json', specification)
    temporary = cache.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(cache)
    return root, payload, specification


def license_check(store, threads):
    with model('full_model_license_check', 10, threads) as m:
        x = m.addMVar(2101, lb=0, ub=1)
        m.addConstr(x.sum() >= 1)
        m.setObjective(x @ x)
        result = optimize(m, store, 'license_preflight')
    return result


def seeds_for(days, store, args):
    seeds = {}
    for i, day in enumerate(days):
        booked = solve_day(day, day.booked, store, seconds=args.seconds, threads=args.threads, tie=args.tie)
        actual = solve_day(day, day.actual, store, seconds=args.seconds, threads=args.threads,
                           tie=args.tie, warm_start=booked.get('assignment'))
        seeds[day.key] = {'booked': booked, 'actual': actual}
        if i % 25 == 0 or i + 1 == len(days):
            LOG.info('Seed plans: %s, %d/%d days', day.group, i + 1, len(days))
    return seeds


def benchmark(day, alpha, h, seeds, store, args):
    key = f'benchmark:{day.key}:{alpha}:{h}'
    saved = store.get(key)
    if (saved and saved.get('lower') is not None and saved.get('upper') is not None
            and saved['lower'] <= saved['upper'] + 1e-6 and not args.refine_oracles):
        return saved
    result = response_oracle(day, alpha, h, seeds['booked'], seeds['actual'], store,
                             seconds=args.oracle_seconds, threads=args.threads)
    store.put(key, result)
    return result


def pilot(payload, store, args, root):
    rows = []
    for group, bundle in payload['groups'].items():
        ordered = sorted(bundle['days']['train'], key=lambda d: (sum(map(len, d.eligible)), len(d.booked), d.key))
        indices = sorted(set(np.quantile(np.arange(len(ordered)), [0, .5, .9, 1]).astype(int)))
        for index in indices:
            day = ordered[index]
            seeds = {}
            for name, duration in [('booked', day.booked), ('actual', day.actual)]:
                start = time.monotonic()
                result = solve_day(day, duration, store, seconds=args.seconds, threads=args.threads, tie=args.tie)
                seeds[name] = result
                rows.append({'pilot': 'P1', 'day': day.key, 'duration': name,
                             'complete': result['complete'], 'seconds_this_call': time.monotonic() - start,
                             'solve_seconds': result['seconds'], 'result': result})
            for alpha, h in SCENARIOS:
                start = time.monotonic()
                result = benchmark(day, alpha, h, seeds, store, args)
                rows.append({'pilot': 'P2', 'day': day.key, 'alpha': alpha, 'h': h,
                             'seconds_this_call': time.monotonic() - start, **result})
            write_json(root / 'pilots_P1_P2.json', rows)
            LOG.info('Pilots P1/P2: %s', day.key)
    return rows


def training_checks(payload, store, args, root):
    all_seeds, oracle_rows, daily_history, history_summary = {}, [], [], []
    pending = []
    for group, bundle in payload['groups'].items():
        days = bundle['days']['train']
        seeds = seeds_for(days, store, args)
        all_seeds[group] = seeds
        for day in days:
            if not all(p['complete'] for p in seeds[day.key].values()):
                pending.append(day.key)
                continue
            for alpha, h in SCENARIOS:
                result = benchmark(day, alpha, h, seeds[day.key], store, args)
                if (result.get('lower') is None or result.get('upper') is None
                        or result['lower'] > result['upper'] + 1e-6):
                    pending.append(f'{day.key}:oracle_{alpha}_{h}')
                oracle_rows.append({'day': day.key, 'group': group, 'alpha': alpha, 'h': h,
                                    'booked_cost': metrics(day, seeds[day.key]['booked']['assignment'], day.actual)['cost'],
                                    'actual_oracle': seeds[day.key]['actual']['primary_cost'],
                                    'lower': result['lower'], 'upper': result['upper'],
                                    'complete': result['complete']})
            if len(oracle_rows) % 100 == 0:
                LOG.info('Training benchmarks: %s', day.key)
        rows, summary = historical_checks(days, seeds)
        daily_history.extend(rows); history_summary.extend(summary)
    directory = root / 'checks'
    directory.mkdir(exist_ok=True)
    pd.DataFrame(oracle_rows).to_csv(directory / 'training_oracles.csv', index=False)
    pd.DataFrame(daily_history).to_csv(directory / 'historical_daily.csv', index=False)
    pd.DataFrame(history_summary).to_csv(directory / 'historical_summary.csv', index=False)
    write_json(directory / 'status.json', {'ready': not pending, 'pending_checks': pending,
                                         'oracle_brackets_allowed': True})
    return all_seeds, pending


def baselines(payload, store, args, root):
    ready = True
    for group, bundle in payload['groups'].items():
        frame, X, days = bundle['frames']['train'], bundle['X']['train'], bundle['days']['train']
        b, a = frame.booked.to_numpy(), frame.actual.to_numpy()
        state = store.get('models:' + group) or {'scenarios': {}}
        if 'penalty' not in state:
            state['penalty'] = penalty(X, len(days), args.seed)
        lam1 = state['penalty']['lambda1']
        if not state.get('direct_case', {}).get('complete'):
            state['direct_case'] = fit_policy(X, b, a, len(days), 1., None, lam1, np.zeros(X.shape[1]),
                                              store, group + '_direct_case', seconds=args.seconds, threads=args.threads)
            store.put('models:' + group, state)
        if not state['direct_case']['complete']:
            ready = False
            continue
        for alpha, h in SCENARIOS:
            key = scenario_key(alpha, h)
            scenario = state['scenarios'].setdefault(key, {})
            if not scenario.get('Shift', {}).get('complete'):
                scenario['Shift'] = train_shift(days, alpha, h, store, seconds=args.seconds,
                                                threads=args.threads, tie=args.tie, progress=LOG.info)
                store.put('models:' + group, state)
            if not scenario.get('Case-Error', {}).get('complete'):
                start = project(np.asarray(state['direct_case']['w']) / alpha, X, b)
                scenario['Case-Error'] = fit_policy(X, b, a, len(days), alpha, h, alpha * lam1,
                                                   start, store, group + '_Case-Error_' + key,
                                                   seconds=args.seconds, threads=args.threads)
                store.put('models:' + group, state)
            ready &= scenario['Shift']['complete'] and scenario['Case-Error']['complete']
            LOG.info('Baselines: %s %s', group, key)
        write_json(root / f'models_{group}.json', state)
    return ready


def vf_stage(payload, seeds, store, args, root, pilot_only=False):
    ready, timings = True, []
    for group, bundle in payload['groups'].items():
        state = store.get('models:' + group)
        if not state or not state.get('direct_case', {}).get('complete'):
            ready = False
            continue
        frame, X, days = bundle['frames']['train'], bundle['X']['train'], bundle['days']['train']
        b, a, lam1 = frame.booked.to_numpy(), frame.actual.to_numpy(), state['penalty']['lambda1']
        jobs = [(1., None, 'VF-Direct', state['direct_case'])]
        jobs += [(alpha, h, 'VF', state['scenarios'].get(scenario_key(alpha, h), {}).get('Case-Error', {}))
                 for alpha, h in SCENARIOS]
        if pilot_only:
            jobs = [job for job in jobs if job[:2] == (.8, 60)]
        for alpha, h, method, initial in jobs:
            previous_pilot = store.get('pilot_P3:' + group) if pilot_only else None
            if previous_pilot and previous_pilot['complete']:
                timings.append(previous_pilot)
                continue
            if not initial.get('complete'):
                ready = False
                continue
            target = state if h is None else state['scenarios'][scenario_key(alpha, h)]
            if not pilot_only and target.get(method, {}).get('complete'):
                continue
            tag = group + '_' + method + '_' + str(h) + '_' + str(alpha)
            start = time.monotonic()
            result = train_vf(X, b, a, days, alpha, h, alpha * lam1, initial['w'], seeds[group], store,
                              ('P3_' if pilot_only else '') + tag, seconds=args.seconds,
                              threads=args.threads, tie=args.tie, max_outer=1 if pilot_only else 15,
                              progress=LOG.info)
            timing = {'group': group, 'seconds': time.monotonic() - start +
                      (previous_pilot['seconds'] if previous_pilot else 0.),
                      'complete': result['complete'], 'outer_iterations': len(result['trajectory'])}
            timings.append(timing)
            if pilot_only:
                store.put('pilot_P3:' + group, timing)
            ready &= result['complete']
            if not pilot_only:
                target[method] = result
                store.put('models:' + group, state)
                write_json(root / f'models_{group}.json', state)
        if pilot_only:
            write_json(root / 'pilot_P3.json', timings)
    return ready


def policy_snapshot(payload, store):
    snapshot, diagnostics = {}, []
    for group in GROUPS:
        state = store.get('models:' + group)
        if not state or not state.get('VF-Direct', {}).get('complete'):
            return None, []
        snapshot[group] = {'VF-Direct': state['VF-Direct']['w'], 'scenarios': {}}
        fits = [('direct_case', None, state['direct_case']), ('VF-Direct', None, state['VF-Direct'])]
        for alpha, h in SCENARIOS:
            key = scenario_key(alpha, h)
            models = state['scenarios'].get(key, {})
            if any(not models.get(method, {}).get('complete') for method in ('Shift', 'Case-Error', 'VF')):
                return None, []
            snapshot[group]['scenarios'][key] = {'Shift': models['Shift']['shift'],
                                                **{method: models[method]['w'] for method in ('Case-Error', 'VF')}}
            fits.extend((method, key, models[method]) for method in ('Case-Error', 'VF'))
        for method, scenario, fit in fits:
            diagnostics.append({'group': group, 'method': method, 'scenario': scenario,
                                'max_abs_coefficient': fit['max_abs_coefficient'],
                                'coefficient_bound_binds': fit['coefficient_bound_binds'],
                                'hit_outer_cap': fit.get('hit_outer_cap', False),
                                'final_bound': fit.get('final_bound'), 'zero_bound': fit.get('zero_bound')})
    return snapshot, diagnostics


def save_training_diagnostics(root, payload, diagnostics):
    write_json(root / 'training_diagnostics.json', diagnostics)
    rows = []
    for fit in diagnostics:
        if fit['final_bound'] is None:
            continue
        days = len(payload['groups'][fit['group']]['days']['train'])
        for policy, field in [('learned', 'final_bound'), ('zero', 'zero_bound')]:
            bound = fit[field]
            rows.append({'group': fit['group'], 'method': fit['method'],
                         'scenario': fit['scenario'] or 'direct', 'policy': policy,
                         'days': days, 'bound_sum': bound['bound_sum'],
                         'regularized_bound_sum': days * bound['regularized_bound']})
    table = pd.DataFrame(rows)
    if not table.empty:
        totals = table.groupby(['method', 'scenario', 'policy'], as_index=False)[
            ['days', 'bound_sum', 'regularized_bound_sum']].sum()
        totals['group'] = 'All'
        table = pd.concat([table, totals], ignore_index=True)
        table['bound_mean'] = table.bound_sum / table.days
        table['regularized_bound_mean'] = table.regularized_bound_sum / table.days
        table.to_csv(root / 'training_certificates.csv', index=False)


def evaluate(payload, store, args, root):
    policies, diagnostics = policy_snapshot(payload, store)
    if policies is None:
        return {'ready': False, 'reason': 'training is incomplete; no test policies evaluated'}
    fingerprint = digest(policies)
    frozen = store.get('test_freeze')
    if frozen and frozen['fingerprint'] != fingerprint:
        return {'ready': False, 'reason': 'policies differ from the frozen test run'}
    if not frozen:
        frozen = {'fingerprint': fingerprint, 'policies': policies}
        store.put('test_freeze', frozen)
        write_json(root / 'frozen_test_policies.json', frozen)
    save_training_diagnostics(root, payload, diagnostics)
    daily, oracles, expected = [], [], []
    for group, bundle in payload['groups'].items():
        frame, X, days = bundle['frames']['test'], bundle['X']['test'], bundle['days']['test']
        expected.extend(days)
        b = frame.booked.to_numpy()
        seeds = seeds_for(days, store, args)
        for alpha, h in SCENARIOS:
            policy = policies[group]['scenarios'][scenario_key(alpha, h)]
            for method in METHODS:
                if method == 'Booked':
                    raw = np.zeros(len(b))
                elif method == 'Shift':
                    raw = np.full(len(b), policy[method] / alpha)
                else:
                    weights = policies[group][method] if method == 'VF-Direct' else policy[method]
                    raw = X @ np.asarray(weights)
                _, duration = display_and_duration(raw, b, alpha, h)
                for day in days:
                    plan = seeds[day.key]['booked'] if method == 'Booked' else solve_day(
                        day, duration[day.rows], store, seconds=args.seconds, threads=args.threads, tie=args.tie,
                        warm_start=seeds[day.key]['booked'].get('assignment'))
                    row = {'day': day.key, 'date': day.date, 'group': group, 'alpha': alpha, 'h': h,
                           'method': method, 'complete': plan['complete'], 'cases': len(day.booked)}
                    if plan['complete']:
                        row.update(metrics(day, plan['assignment'], day.actual))
                        row['absolute_error'] = float(np.abs(day.actual - duration[day.rows]).sum())
                        row.pop('loads')
                    daily.append(row)
                    store.put(f'evaluation:{day.key}:{alpha}:{h}:{method}', row)
                LOG.info('Test evaluation: %s alpha=%s h=%s %s', group, alpha, h, method)
            for day in days:
                result = benchmark(day, alpha, h, seeds[day.key], store, args)
                oracles.append({'day': day.key, 'date': day.date, 'group': group, 'alpha': alpha, 'h': h,
                                'actual_oracle': seeds[day.key]['actual'].get('primary_cost'),
                                'lower': result['lower'], 'upper': result['upper'], 'complete': result['complete']})
    return summarize(root / 'report', daily, oracles, expected, args.seed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workbook', type=Path)
    parser.add_argument('--out', type=Path, default=Path('results'))
    parser.add_argument('--stage', choices=['audit', 'pilot', 'checks', 'baselines', 'pilot-vf', 'train', 'evaluate', 'all'], default='all')
    parser.add_argument('--seconds', type=float, default=300, help='Budget per daily solve or convex update; pending work is resumable')
    parser.add_argument('--oracle-seconds', type=float, default=30, help='Budget per response-limited oracle day/scenario')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--tie', choices=['squares', 'lexloads'], default='squares')
    parser.add_argument('--seed', type=int, default=20261007)
    parser.add_argument('--refine-oracles', action='store_true', help='Spend another budget on unfinished oracle brackets')
    args = parser.parse_args(argv)
    if args.seconds <= 0 or args.oracle_seconds <= 0 or args.threads < 1:
        parser.error('Time budgets and thread count must be positive')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    root, payload, specification = prepare(args.workbook, args.out, args.tie, args.seed)
    LOG.info('Run directory: %s', root.resolve())
    store = Store(root)
    status = {'stage': args.stage, 'ready': False}
    try:
        if args.stage == 'audit':
            status['ready'] = True
            return 0
        if not payload['audit']['booking_grid_verified']:
            status['reason'] = 'The raw booking grid does not support the specified +1 correction'
            return 2
        if args.stage in ('all', 'baselines', 'pilot-vf', 'train'):
            check = license_check(store, args.threads)
            if not check['optimal']:
                status['reason'] = 'Full-size solver preflight failed; no learning started'
                status['solver'] = check
                LOG.error('%s: %s', status['reason'], check.get('error', check['status']))
                return 2
        if args.stage in ('all', 'pilot', 'checks', 'baselines', 'pilot-vf', 'train'):
            rows = pilot(payload, store, args, root)
            pending_p1 = [r['day'] for r in rows if r['pilot'] == 'P1' and not r['complete']]
            if pending_p1:
                status.update(reason='P1 has unfinished daily plans; resume with a larger solve budget',
                              pending_days=sorted(set(pending_p1)))
                return 2
            if args.stage == 'pilot':
                status['ready'] = True
                return 0
        if args.stage == 'evaluate':
            status.update(evaluate(payload, store, args, root))
            return 0 if status['ready'] else 2
        seeds, pending = training_checks(payload, store, args, root)
        if pending:
            status.update(reason='Training checks pending; inspect checks/status.json and resume', pending_days=pending)
            return 2
        if args.stage == 'checks':
            status['ready'] = True
            return 0
        if not baselines(payload, store, args, root):
            status['reason'] = 'Baseline fitting or exact Shift plans pending; progress is saved'
            return 2
        if args.stage == 'baselines':
            status['ready'] = True
            return 0
        if args.stage in ('all', 'pilot-vf', 'train'):
            if not vf_stage(payload, seeds, store, args, root, pilot_only=True):
                status['reason'] = 'P3 pending; progress is saved'
                return 2
            if args.stage == 'pilot-vf':
                status['ready'] = True
                return 0
        if not vf_stage(payload, seeds, store, args, root):
            status['reason'] = 'VF training pending; progress is saved'
            return 2
        _, diagnostics = policy_snapshot(payload, store)
        save_training_diagnostics(root, payload, diagnostics)
        if args.stage == 'all':
            status.update(evaluate(payload, store, args, root))
        else:
            status['ready'] = True
        return 0 if status['ready'] else 2
    except KeyboardInterrupt:
        status['reason'] = 'Interrupted; completed day and policy checkpoints are saved'
        return 130
    except (gp.GurobiError, OSError) as exc:
        status['reason'] = str(exc)
        LOG.error('Run paused: %s', exc)
        return 2
    finally:
        write_json(root / 'status.json', status)
        store.export_log(root / 'solver_log.jsonl')
        store.close()
        LOG.info('Stage %s: %s. %s', args.stage, 'ready' if status['ready'] else 'pending', status.get('reason', ''))


if __name__ == '__main__':
    sys.exit(main())
