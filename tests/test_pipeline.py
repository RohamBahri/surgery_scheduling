from argparse import Namespace

import numpy as np
import pandas as pd

from surgery.data import GROUPS, Encoder, build_days
from surgery.experiment import baselines, evaluate, policy_snapshot, training_checks, vf_stage
from surgery.learning import penalty
from surgery.storage import Store


def test_penalty_uses_training_days_and_fixed_seed():
    X = np.column_stack([np.ones(10), np.linspace(-1, 1, 10)])
    first = penalty(X, 2, seed=17)
    assert first == penalty(X, 2, seed=17)
    assert first['lambda1'] == 2 * penalty(X, 4, seed=17)['lambda1']
    assert first['draws'] == 1000 and first['tau'] == '7/11'


def test_complete_pipeline_and_identical_test_resume(tmp_path):
    payload = {'groups': {}}
    for group in GROUPS:
        train = pd.DataFrame({'case_id': [2, 3], 'group': [group] * 2,
                              'date': pd.to_datetime(['2011-07-04'] * 2), 'service': ['s'] * 2,
                              'procedure': ['p'] * 2, 'surgeon': ['a', 'b'], 'room': ['A', 'B'],
                              'booked': [120., 200.], 'actual': [140., 200.]})
        test = train.copy(); test.date = pd.Timestamp('2013-01-07')
        encoder = Encoder()
        X = encoder.fit_transform(train)
        payload['groups'][group] = {'frames': {'train': train, 'test': test},
                                    'X': {'train': X, 'test': encoder.transform(test)},
                                    'days': {'train': build_days(train, train), 'test': build_days(test, train)}}
    store = Store(tmp_path)
    args = Namespace(seconds=5, oracle_seconds=5, threads=1, tie='squares', seed=7, refine_oracles=False)
    seeds, pending = training_checks(payload, store, args, tmp_path)
    assert not pending
    assert baselines(payload, store, args, tmp_path)
    assert vf_stage(payload, seeds, store, args, tmp_path, pilot_only=True)
    assert vf_stage(payload, seeds, store, args, tmp_path)
    policies, diagnostics = policy_snapshot(payload, store)
    assert policies and diagnostics
    first = evaluate(payload, store, args, tmp_path)
    assert first['ready']
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
