import numpy as np
import pandas as pd

from surgery.data import COLUMNS, Encoder, build_cohort, build_days, overlap_flags, pairwise_spread


def raw_case(day='2011-07-04', room='OR1', surgeon='a', rin='08:00:00', **changes):
    row = {c: None for c in COLUMNS}
    row.update(Operating_Room=room, Surgeon_Code=surgeon, Main_Procedure_Id='p', Case_Service='s',
               Patient_ID='patient', Decision_Date=pd.Timestamp(day) - pd.Timedelta(days=2),
               Patient_Type='ELECTIVE')
    row['Booked Time (Minutes)'] = 59
    for stem, clock in [('Enter Room', rin), ('Actual Start', '08:10:00'),
                        ('Actual Stop', '08:40:00'), ('Leave Room', '09:00:00')]:
        row[stem + ' Date'] = pd.Timestamp(day)
        row[stem + ' Time'] = clock
    row.update(changes)
    return row


def test_contamination_scope_group_key_and_booking_correction():
    cases = [raw_case(room='OR101', surgeon='x'), raw_case(room='DS101', surgeon='x'),
             raw_case(room='OR102', surgeon='x', **{'Actual Stop Time': '08:10:00'}),
             raw_case(room='OR1', surgeon='y'),
             raw_case(room='OR2', surgeon='y', rin='20:00:00', Patient_Type='EMERGENCY'),
             raw_case(room='OR3', surgeon='z', **{'Booked Time (Minutes)': 599}),
             raw_case(room='PMH01', surgeon='p')]
    cohort, audit, excluded, _ = build_cohort(pd.DataFrame(cases))
    assert set(cohort.case_id) == {3, 5, 7}
    assert set(cohort.booked) == {60, 600}
    assert not cohort.group.eq('PMH').any()
    assert excluded.loc[excluded.case_id.eq(2), 'rule'].item() == 'complete_group_surgeon_days'
    assert audit['booking_grid_verified']


def test_nested_intervals_and_strict_overlap_threshold():
    day = pd.Timestamp('2011-07-04')
    rows = pd.DataFrame({'room': ['OR1'] * 4, 'date': [day] * 4,
                         'rin': [day + pd.Timedelta(minutes=x) for x in [0, 10, 60, 105]],
                         'rout': [day + pd.Timedelta(minutes=x) for x in [120, 30, 80, 130]]})
    assert overlap_flags(rows).tolist() == [True, True, True, False]


def score_frame():
    dates = pd.to_datetime(['2011-07-04'] * 4 + ['2011-07-05'] * 4 + ['2011-07-06'] * 4)
    return pd.DataFrame({'date': dates, 'service': ['s'] * 12,
                         'procedure': ['a', 'a', 'b', 'b'] * 3,
                         'surgeon': ['x', 'x', 'y', 'y'] * 3, 'booked': [60] * 12,
                         'actual': [59, 61, 79, 81, 58, 62, 78, 82, 50, 70, 90, 100]})


def test_training_scores_leave_out_own_date_and_test_uses_training_only():
    train = score_frame()
    first = Encoder()
    first.fit_transform(train)
    changed = train.copy()
    mask = changed.date.eq(pd.Timestamp('2011-07-04')) & changed.procedure.eq('a')
    changed.loc[mask, 'actual'] += 1000
    second = Encoder()
    second.fit_transform(changed)
    own = train.date.eq(pd.Timestamp('2011-07-04')).to_numpy()
    np.testing.assert_allclose(first.training_scores[own], second.training_scores[own])
    assert np.any(first.training_scores[~own] != second.training_scores[~own])

    test = train.iloc[:2].copy()
    before = first.transform(test)
    test.actual += 5000
    np.testing.assert_allclose(before, first.transform(test))
    test['procedure'], test['surgeon'] = 'unseen', 'unseen'
    assert np.all(first._scores(test) == 0)


def test_pairwise_spread_and_full_training_shrinkage():
    assert pairwise_spread([4]) == 0
    assert pairwise_spread([1, 4]) == 3
    assert np.isclose(pairwise_spread([1, 3, 8]), 14 / 3)

    rows = pd.DataFrame({'date': pd.to_datetime(['2011-07-04'] * 5),
                         'service': ['s'] * 5, 'procedure': ['a', 'a', 'b', 'b', 'b'],
                         'surgeon': ['x', 'x', 'y', 'y', 'y'], 'booked': [60] * 5,
                         'actual': [60, 62, 69, 70, 71]})
    within = 4 / 3
    between = 2 * (1 - 6.4) ** 2 + 3 * (10 - 6.4) ** 2
    expected = within / ((between - within) / 2.4)
    assert np.isclose(Encoder._shrinkage(rows, 'procedure'), expected)
    rows.actual = [59, 61, 59, 60, 61]
    assert Encoder._shrinkage(rows, 'procedure') is None


def test_weekday_eligibility_and_both_fallback_levels():
    train = pd.DataFrame({
        'group': ['TGH'] * 4,
        'date': pd.to_datetime(['2011-07-04', '2011-07-04', '2011-07-05', '2011-07-05']),
        'room': ['OR1', 'OR2', 'OR2', 'OR1'],
        'service': ['A', 'B', 'C', 'B'],
        'surgeon': ['a', 'b', 'c', 'b'],
        'case_id': [2, 3, 4, 5], 'booked': [60.] * 4, 'actual': [60.] * 4})
    test = pd.DataFrame({
        'group': ['TGH'] * 3, 'date': pd.to_datetime(['2013-01-07'] * 3),
        'room': ['OR1', 'OR2', 'OR1'], 'service': ['A', 'C', 'D'],
        'surgeon': ['a', 'c', 'd'], 'case_id': [6, 7, 8],
        'booked': [60.] * 3, 'actual': [60.] * 3})
    day = build_days(test, train)[0]
    assert day.eligibility_source == ('weekday', 'all_week_fallback', 'candidate_room_fallback')
    assert tuple(day.rooms[r] for r in day.eligible[0]) == ('OR1',)
    assert tuple(day.rooms[r] for r in day.eligible[1]) == ('OR2',)
    assert tuple(day.rooms[r] for r in day.eligible[2]) == ('OR1', 'OR2')
    assert day.weekday_fallback_surgeons == 2
    assert day.fallback_surgeons == 1
