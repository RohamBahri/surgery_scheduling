"""Matched test summaries, oracle brackets, and paired calendar-week inference."""
from pathlib import Path

import numpy as np
import pandas as pd

from .data import GROUPS, SCENARIOS
from .planner import metrics
from .storage import write_json

METHODS = ('Booked', 'Shift', 'Case-Error', 'VF-Direct', 'VF')


def opportunity_interval(booked, cost, lower, upper):
    numerator = booked - cost
    upper = min(upper, booked)
    big, small = booked - lower, booked - upper
    if big <= 1e-6:
        return None, None, 'zero_or_unresolved_opportunity'
    if small <= 1e-6:
        if numerator < -1e-6:
            return None, numerator / big, 'denominator_may_be_zero'
        return None, None, 'denominator_may_be_zero'
    values = numerator / big, numerator / small
    return min(values), max(values), 'bounded'


def summarize(root, daily, oracles, expected, seed=20261007):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    data, oracle = pd.DataFrame(daily), pd.DataFrame(oracles)
    data.to_csv(root / 'daily_methods.csv', index=False)
    oracle.to_csv(root / 'daily_oracles.csv', index=False)
    required = {(d.key, a, h, method) for d in expected for a, h in SCENARIOS for method in METHODS}
    finished = {(r['day'], r['alpha'], r['h'], r['method']) for r in daily if r['complete']}
    required_oracles = {(d.key, a, h) for d in expected for a, h in SCENARIOS}
    bounded = {(r['day'], r['alpha'], r['h']) for r in oracles
               if r.get('lower') is not None and r.get('upper') is not None
               and r['lower'] <= r['upper'] + 1e-6}
    status = {'ready': (required == finished and len(daily) == len(required)
                        and required_oracles == bounded and len(oracles) == len(required_oracles)),
              'missing_method_days': len(required - finished),
              'missing_or_invalid_oracle_brackets': len(required_oracles - bounded),
              'unfinished_oracle_days': sum(not r['complete'] for r in oracles),
              'bootstrap_seed': seed, 'bootstrap_resamples': 10000}
    if status['ready']:
        keys = ['day', 'alpha', 'h']
        witness = data.groupby(keys).cost.min().rename('method_upper')
        oracle = oracle.join(witness, on=keys)
        oracle['solver_upper'] = oracle.upper
        oracle['upper'] = np.minimum(oracle.upper, oracle.method_upper)
        inconsistent = oracle.lower > oracle.upper + 1e-6
        status['ready'] = not bool(inconsistent.any())
        status['missing_or_invalid_oracle_brackets'] += int(inconsistent.sum())
        oracle.to_csv(root / 'daily_oracles.csv', index=False)
    write_json(root / 'status.json', status)
    if not status['ready']:
        (root / 'README.md').write_text(
            'Evaluation is incomplete. Daily checkpoints are saved. Resume evaluation to fill the missing '
            'method days or oracle brackets; publication tables are not produced from a selected subset.\n')
        return status
    data['week'] = pd.to_datetime(data.date).dt.to_period('W-SUN').dt.start_time
    calendar = pd.date_range('2012-12-31', '2013-06-24', freq='W-MON')
    rng = np.random.default_rng(seed)
    resamples = rng.integers(0, len(calendar), size=(10000, len(calendar)))
    np.save(root / 'bootstrap_week_indices.npy', resamples)
    pd.DataFrame({'week_start': calendar}).to_csv(root / 'bootstrap_weeks.csv', index=False)
    summaries, comparisons = [], []
    for alpha, h in SCENARIOS:
        scenario = data[data.alpha.eq(alpha) & data.h.eq(h)]
        os = oracle[oracle.alpha.eq(alpha) & oracle.h.eq(h)]
        for group in (*GROUPS, 'All'):
            rows = scenario if group == 'All' else scenario[scenario.group.eq(group)]
            bounds = os if group == 'All' else os[os.group.eq(group)]
            booked = rows[rows.method.eq('Booked')].cost.sum()
            lower, upper = bounds.lower.sum(), bounds.upper.sum()
            for method in METHODS:
                selected = rows[rows.method.eq(method)]
                cost = selected.cost.sum()
                lo, hi, reason = opportunity_interval(booked, cost, lower, upper)
                summaries.append({
                    'alpha': alpha, 'h': h, 'group': group, 'method': method,
                    'cost': cost, 'saving_vs_booked': booked - cost,
                    'saving_vs_booked_pct': 100 * (booked - cost) / booked if booked else None,
                    'overtime': selected.overtime.sum(), 'idle': selected.idle.sum(),
                    'room_days': int(selected.rooms.sum()), 'cases': int(selected.cases.sum()),
                    'post_review_mae': selected.absolute_error.sum() / selected.cases.sum(),
                    'realized_oracle': bounds.actual_oracle.sum(),
                    'response_oracle_lower': lower, 'response_oracle_upper': upper,
                    'response_oracle_exact': bool(bounds.complete.all()),
                    'opportunity_captured_lower': lo, 'opportunity_captured_upper': hi,
                    'opportunity_ratio_status': reason})
            weekly = rows.groupby(['week', 'method']).cost.sum().unstack(fill_value=0).reindex(calendar, fill_value=0)
            for comparator in ('Case-Error', 'Shift', 'VF-Direct'):
                difference = (weekly.VF - weekly[comparator]).to_numpy()
                reference = weekly[comparator].to_numpy()
                means = difference[resamples].mean(axis=1)
                denominators = reference[resamples].sum(axis=1)
                percentages = np.divide(100 * difference[resamples].sum(axis=1), denominators,
                                        out=np.full(len(resamples), np.nan), where=denominators > 0)
                interval = np.quantile(means, [.025, .975])
                percentage_interval = np.nanquantile(percentages, [.025, .975]) if np.isfinite(percentages).any() else [np.nan, np.nan]
                comparisons.append({
                    'alpha': alpha, 'h': h, 'group': group, 'comparator': comparator,
                    'mean_weekly_vf_minus_comparator': float(difference.mean()),
                    'difference_pct_of_comparator': 100 * difference.sum() / reference.sum() if reference.sum() else None,
                    'ci95_weekly_lower': interval[0], 'ci95_weekly_upper': interval[1],
                    'ci95_pct_lower': percentage_interval[0], 'ci95_pct_upper': percentage_interval[1],
                    'weeks_favoring_vf': int((difference < -1e-6).sum()),
                    'weeks_tied': int((np.abs(difference) <= 1e-6).sum()), 'weeks': len(calendar)})
    pd.DataFrame(summaries).to_csv(root / 'methods.csv', index=False)
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(root / 'comparisons.csv', index=False)
    directions = []
    for (group, comparator), rows in comparison.groupby(['group', 'comparator']):
        signs = np.sign(rows.mean_weekly_vf_minus_comparator)
        directions.append({'group': group, 'comparator': comparator,
                           'vf_lower_in_all_four': bool((signs < 0).all()),
                           'vf_higher_in_all_four': bool((signs > 0).all())})
    write_json(root / 'scenario_directions.json', directions)
    (root / 'README.md').write_text(
        'Read results in this order: Booked versus realized-duration oracle; Booked versus '
        'response-limited oracle brackets; opportunity captured; VF versus Case-Error, Shift, and VF-Direct.\n\n'
        'Positive savings mean lower cost than Booked. Negative comparison differences favor VF. '
        'Percentages use the comparator total cost. All groups and methods use the same 10,000 '
        'calendar-week resamples, including weeks with zero retained activity in a group. '
        'The opportunity ratio uses both oracle bounds; each evaluated feasible method also supplies '
        'a valid upper bound. The best daily witness supplies one common bracket for every method. Undefined ratios are empty, not zero.\n\n'
        'A direction shared by all four scenarios applies to these four modeled scenarios only. '
        'These are retained elective workloads, not complete hospital workloads or measured cash savings.\n')
    return status


def historical_checks(days, seeds):
    rows = []
    for day in days:
        constrained = []
        for surgeon, eligible in enumerate(day.eligible):
            history = day.historical[day.case_surgeon == surgeon]
            constrained.append(min(eligible, key=lambda room: (-int((history == room).sum()), room)))
        for name, assignment, case_assignment in [
            ('Historical', day.historical, True), ('Historical-one-room', constrained, False),
            ('Booked', seeds[day.key]['booked'].get('assignment'), False)]:
            if assignment is None or (name == 'Booked' and not seeds[day.key]['booked']['complete']):
                continue
            row = metrics(day, assignment, day.actual, case_assignment=case_assignment)
            rows.append({'group': day.group, 'date': day.date, 'method': name, **row})
    summary = []
    table = pd.DataFrame(rows)
    for (group, method), part in table.groupby(['group', 'method']):
        loads = np.concatenate(part.loads.to_list())
        summary.append({'group': group, 'method': method, 'days': len(part),
                        **{k: float(part[k].sum()) for k in ['cost', 'overtime', 'idle', 'rooms']},
                        **{f'load_p{p}': float(np.percentile(loads, p)) for p in [50, 90, 95, 99]},
                        'load_max': float(loads.max())})
    return rows, summary
