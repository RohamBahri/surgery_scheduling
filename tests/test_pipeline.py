from argparse import Namespace

import numpy as np
import pandas as pd

from surgery.data import GROUPS, Encoder, build_days
from surgery.experiment import baselines, evaluate, policy_snapshot, training_checks, vf_stage
from surgery.learning import penalty
from surgery.storage import Store


def test_penalty_uses_training_days_and_fixed_seed():
    X = np.column_stack([np.ones(10), np.linspace(-1, 1, 10)])
    errors = np.linspace(-2, 2, 10)
    first = penalty(X, 2, seed=17, errors=errors)
    assert first == penalty(X, 2, seed=17, errors=errors)
    assert len(first['zero_pull_over_lambda']) == 1
    assert first['lambda1'] == 2 * penalty(X, 4, seed=17)['lambda1']
    assert first['draws'] == 1000 and first['tau'] == '7/11'


def test_complete_pipeline_and_identical_test_resume(tmp_path, monkeypatch):
    payload = {'groups': {}}
    for group in GROUPS:
        train = pd.DataFrame({'case_id': [2, 3], 'group': [group] * 2,
                              'date': pd.to_datetime(['2011-07-04'] * 2), 'service': ['s'] * 2,
                              'procedure': ['p'] * 2, 'surgeon': ['a', 'b'], 'room': ['A', 'B'],
                              'patient_id': ['p1', 'p2'],
                              'booked': [120., 200.], 'actual': [140., 200.]})
        test = train.copy(); test.date = pd.Timestamp('2013-01-07')
        encoder = Encoder()
        X = encoder.fit_transform(train)
        payload['groups'][group] = {'frames': {'train': train, 'test': test},
                                    'X': {'train': X, 'test': encoder.transform(test)},
                                    'days': {'train': build_days(train, train), 'test': build_days(test, train)},
                                    'feature_names': encoder.names}
    store = Store(tmp_path)
    args = Namespace(seconds=5, oracle_seconds=5, threads=1, workers=1, tie='maxload', seed=7, refine_oracles=False)
    from surgery import experiment
    benchmark = experiment.benchmark
    def forbidden(*args, **kwargs):
        raise AssertionError('Training checks must not launch response oracles')
    monkeypatch.setattr(experiment, 'benchmark', forbidden)
    seeds, pending = training_checks(payload, store, args, tmp_path)
    monkeypatch.setattr(experiment, 'benchmark', benchmark)
    assert not pending
    assert baselines(payload, store, args, tmp_path)
    assert vf_stage(payload, seeds, store, args, tmp_path, pilot_only=True)
    pilot_fits = [r for r in store.solve_rows() if r['label'] == 'policy_convex' and 'VF_60_0.8_outer_1' in r.get('fit', '')]
    assert vf_stage(payload, seeds, store, args, tmp_path)
    assert [r for r in store.solve_rows() if r['label'] == 'policy_convex' and 'VF_60_0.8_outer_1' in r.get('fit', '')] == pilot_fits
    policies, diagnostics = policy_snapshot(payload, store)
    assert policies and diagnostics
    assert all('hit_inner_cap' in r and 'final_library_size' in r and 'rare_services' in r for r in diagnostics)
    first = evaluate(payload, store, args, tmp_path)
    assert first['ready']
    case_results = pd.read_csv(tmp_path / 'report' / 'case_results.csv')
    assert len(case_results) == len(GROUPS) * 2 * 4 * 5
    assert not case_results.duplicated(['group', 'case_id', 'alpha', 'h', 'method']).any()
    rooms = pd.read_csv(tmp_path / 'report' / 'room_day_results.csv')
    calculated = (case_results.groupby(['group', 'date', 'alpha', 'h', 'method', 'assigned_room'])
                  .planned_duration.agg(['sum', 'count']).reset_index())
    calculated['expected_load'] = calculated['sum'] + 30 * (calculated['count'] - 1)
    merged = rooms.merge(calculated,
                         left_on=['group', 'date', 'alpha', 'h', 'method', 'room'],
                         right_on=['group', 'date', 'alpha', 'h', 'method', 'assigned_room'])
    assert np.allclose(merged.planned_load, merged.expected_load)
    solve_count = len(store.solve_rows())
    second = evaluate(payload, store, args, tmp_path)
    assert first == second
    assert len(store.solve_rows()) == solve_count
    state = store.get('models:TGH')
    state['VF-Direct']['w'][0] += 1
    store.put('models:TGH', state)
    assert not evaluate(payload, store, args, tmp_path)['ready']
    assert len(store.solve_rows()) == solve_count
    store.close()


def test_parallel_day_shift_matches_serial_and_resumes(tmp_path):
    from test_optimization import make_day
    from surgery.learning import train_shift
    day = make_day([5., 200., 330.], [20., 220., 300.])
    second = make_day([90., 210., 170.], [60., 210., 180.]); second.key += '_2'
    store = Store(tmp_path)
    serial = train_shift([day, second], .5, 4, store, workers=1)
    count = len(store.solve_rows())
    parallel = train_shift([day, second], .5, 4, store, workers=2)
    assert serial == parallel and len(store.solve_rows()) == count
    other_store = Store(tmp_path / 'parallel')
    assert train_shift([day, second], .5, 4, other_store, workers=2) == serial
    assert len(other_store.solve_rows()) > 0
    store.close(); other_store.close()


def test_worker_failure_restarts_only_remaining_jobs(tmp_path, monkeypatch):
    from surgery import storage
    from concurrent.futures.process import BrokenProcessPool
    from surgery.planner import plan_job
    from test_optimization import make_day
    attempts = []
    class Pool:
        def __init__(self, max_workers, mp_context):
            attempts.append(max_workers)
        def shutdown(self, **kwargs):
            pass
        def map(self, function, jobs):
            for index, (_, items, _) in enumerate(jobs):
                if len(attempts) == 1 and index == 1:
                    raise BrokenProcessPool('injected worker failure')
                yield [(i, args[0].key) for i, args in items]
    monkeypatch.setattr(storage, 'ProcessPoolExecutor', Pool)
    days = [make_day([100.]) for _ in range(3)]
    for i, day in enumerate(days):
        day.key = str(i)
    store = Store(tmp_path)
    jobs = [(day, day.booked, {}) for day in days]
    assert list(storage.map_days(plan_job, jobs, store, workers=4)) == ['0', '1', '2']
    assert attempts == [4, 2]
    store.close()
