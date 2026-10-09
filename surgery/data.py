"""UHN cohort, offline cross-fitted features, and room-group/day instances."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

GROUPS = ('TGH', 'TWH-main', 'TWH-day')
ROOMS = {
    'TGH': tuple(f'OR{i}' for i in (*range(1, 18), 19, 21)),
    'TWH-main': tuple(f'OR{i}' for i in (*range(101, 112), 114, 115)),
    'TWH-day': tuple(f'DS{i}' for i in range(101, 105)),
}
SCENARIOS = ((0.5, 30), (0.5, 60), (0.8, 30), (0.8, 60))
TIMESTAMPS = {'Enter Room': 'rin', 'Actual Start': 'start',
              'Actual Stop': 'stop', 'Leave Room': 'rout'}
COLUMNS = ['Operating_Room', 'Surgeon_Code', 'Main_Procedure_Id', 'Case_Service',
           'Booked Time (Minutes)', 'Decision_Date', 'Patient_Type', 'Patient_ID',
           'Case_Cancelled_Reason', 'Case Cancel Date'] + [
               stem + suffix for stem in TIMESTAMPS for suffix in (' Date', ' Time')]


def identifier(series):
    return series.fillna('').astype(str).str.strip().str.replace(r'\.0$', '', regex=True)


def pairwise_spread(values):
    """Average |X-X'| over distinct pairs; zero when fewer than two observations exist."""
    values = np.sort(np.asarray(values, float))
    n = len(values)
    if n < 2:
        return 0.
    coefficients = 2 * np.arange(n) - n + 1
    return float(2 * (coefficients @ values) / (n * (n - 1)))


def overlap_flags(frame):
    flagged = pd.Series(False, index=frame.index)
    for _, rows in frame.groupby(['room', 'date'], sort=False):
        start, end = rows.rin.to_numpy(), rows.rout.to_numpy()
        overlap = np.minimum(end[:, None], end) - np.maximum(start[:, None], start)
        bad = (overlap > np.timedelta64(15, 'm')) & ~np.eye(len(rows), dtype=bool)
        flagged.loc[rows.index] = bad.any(axis=1)
    return flagged


def _booking_summary(frame, label):
    x = frame.booked.to_numpy(float)
    q = np.quantile(x, [.01, .05, .25, .5, .75, .95, .99])
    return {'group': label, 'cases': len(x), 'min': float(x.min()),
            **{f'p{p}': float(v) for p, v in zip((1, 5, 25, 50, 75, 95, 99), q)},
            'max': float(x.max()), 'booked_le_15': int((x <= 15).sum()),
            'booked_le_30': int((x <= 30).sum()), 'booked_gt_480': int((x > 480).sum())}


def build_cohort(raw):
    missing = set(COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f'Missing workbook columns: {sorted(missing)}')
    d = pd.DataFrame(index=raw.index)
    d['case_id'] = np.arange(len(raw)) + 2
    d['room'] = raw.Operating_Room.fillna('').str.upper().str.replace(r'\s+', '', regex=True)
    d['group'] = d.room.map({r: g for g, rooms in ROOMS.items() for r in rooms})
    for source, target in [('Surgeon_Code', 'surgeon'), ('Main_Procedure_Id', 'procedure'),
                           ('Case_Service', 'service'), ('Patient_ID', 'patient_id')]:
        d[target] = identifier(raw[source])
    d['patient_type'] = raw.Patient_Type.fillna('').astype(str).str.strip()
    booking = pd.to_numeric(raw['Booked Time (Minutes)'], errors='coerce')
    d['booked'] = booking + 1
    for source, target in TIMESTAMPS.items():
        d[target] = (pd.to_datetime(raw[source + ' Date'], errors='coerce').dt.normalize()
                     + pd.to_timedelta(raw[source + ' Time'], errors='coerce'))
    d['date'] = d.rin.dt.normalize()
    for timestamp in ['start', 'stop', 'rout']:
        d['date'] = d.date.fillna(d[timestamp].dt.normalize())
    d['actual'] = (d.rout - d.rin).dt.total_seconds() / 60
    d['dtt'] = pd.to_datetime(raw.Decision_Date, errors='coerce').dt.normalize()
    d['emergency'] = raw.Patient_Type.fillna('').str.upper().str.contains('EMERGENCY')
    cancel_reason = raw.Case_Cancelled_Reason.fillna('').astype(str).str.strip()
    cancelled = cancel_reason.ne('') | raw['Case Cancel Date'].notna()
    source = d.copy()
    source['cancelled'] = cancelled
    source['cancel_reason'] = cancel_reason
    flow, excluded = [], []

    def retain(mask, rule):
        nonlocal d
        removed = d.loc[~mask, ['case_id', 'group', 'surgeon', 'date']].copy()
        removed['rule'] = rule
        excluded.append(removed)
        d = d.loc[mask].copy()
        flow.append({'rule': rule, 'removed': len(removed), 'remaining': len(d)})

    retain(~cancelled, 'not_cancelled')
    retain(d.group.notna(), 'room_registry_without_PMH')
    mornings = d[d.group.isin(['TGH', 'TWH-main']) & d.rin.dt.hour.ge(7) & d.rin.dt.hour.lt(12)]
    activity = mornings.groupby('date').room.nunique().reindex(
        pd.bdate_range('2011-07-01', '2013-06-30'), fill_value=0)
    median = float(activity.loc[:'2012-12-31'].median())
    closures = activity[activity < 0.25 * median].index
    valid = (d.rin.le(d.start) & d.start.lt(d.stop) & d.stop.le(d.rout)
             & d.actual.gt(0) & d.actual.lt(1440))
    minutes = d.rin.dt.hour * 60 + d.rin.dt.minute + d.rin.dt.second / 60
    other = {
        'positive_raw_booking': d.booked.gt(1) & np.isfinite(d.booked),
        'weekdays': d.date.dt.weekday.lt(5),
        'not_closure': ~d.date.isin(closures),
        'scheduled': ((minutes.ge(420) & minutes.lt(1020))
                      | d.dtt.le(d.date - pd.Timedelta(days=1))),
        'not_emergency': ~d.emergency,
        'study_dates': d.date.between('2011-07-01', '2013-06-30'),
    }
    scope = pd.concat(other, axis=1).all(axis=1)
    eligible = d[scope].copy()
    keys = ['group', 'surgeon', 'date']
    bad_days = d.loc[scope & ~valid, keys].dropna(subset=['date'])
    unknown_dates = int((~valid & d.date.isna()).sum())
    retain(valid, 'ordered_timestamps_and_room_stay')
    for name, mask in other.items():
        retain(mask.loc[d.index], name)
    overlap = overlap_flags(d)
    bad_days = pd.concat([bad_days, d.loc[overlap, keys]]).drop_duplicates()
    retain(~overlap, 'overlap_over_15_minutes')
    retain(~pd.MultiIndex.from_frame(d[keys]).isin(pd.MultiIndex.from_frame(bad_days)),
           'complete_group_surgeon_days')
    if d.empty or d[['surgeon', 'service']].eq('').any().any():
        raise ValueError('Cohort is empty or retained cases lack surgeon/service identifiers')
    d['split'] = np.where(d.date < pd.Timestamp('2013-01-01'), 'train', 'test')
    d = d.sort_values(['group', 'date', 'surgeon', 'case_id']).reset_index(drop=True)

    quality = eligible.groupby(['group', 'date']).agg(
        eligible_cases=('case_id', 'size'), eligible_booked=('booked', 'sum'),
        eligible_rooms=('room', 'nunique'), eligible_surgeons=('surgeon', 'nunique'))
    after = d.groupby(['group', 'date']).agg(
        retained_cases=('case_id', 'size'), retained_booked=('booked', 'sum'),
        retained_rooms=('room', 'nunique'), retained_surgeons=('surgeon', 'nunique'))
    quality = quality.join(after).fillna(0).reset_index()
    quality['damaged'] = quality.eligible_cases.ne(quality.retained_cases)

    cancellation_scope = source[source.group.notna() & source.cancelled].copy()
    cancellation_scope['cancel_reason'] = cancellation_scope.cancel_reason.replace('', 'Cancellation date only')
    cancellation_reasons = [{'reason': str(k), 'cases': int(v)}
                            for k, v in cancellation_scope.cancel_reason.value_counts().items()]

    patients = d[d.patient_id.ne('')]
    patient_counts = patients.groupby('patient_id').size()
    repeat = patient_counts[patient_counts.gt(1)]
    split_counts = patients.groupby('patient_id').split.nunique()

    retained_pairs = set(zip(d.surgeon, d.date))
    retained_ids = set(d.case_id)
    extras = source[~source.cancelled & source.date.notna() & source.surgeon.ne('')].copy()
    extras = extras[[((s, dt) in retained_pairs and case_id not in retained_ids)
                     for s, dt, case_id in zip(extras.surgeon, extras.date, extras.case_id)]]

    training = d[d.split.eq('train')].copy()
    training['weekday'] = training.date.dt.weekday
    block = training.groupby(['group', 'date', 'room', 'service']).size().rename('cases').reset_index()
    totals = block.groupby(['group', 'date', 'room']).cases.transform('sum')
    dominant = block.assign(share=block.cases / totals).groupby(
        ['group', 'date', 'room'], as_index=False).share.max()
    block_summary = []
    for group, rows in dominant.groupby('group'):
        block_summary.append({'group': group, 'room_days': len(rows),
                              'single_service_share': float((rows.share >= 1 - 1e-12).mean()),
                              'dominant_service_ge_80_share': float((rows.share >= .8).mean())})
    weekday_counts = training.groupby(['group', 'room', 'weekday', 'service']).size().rename('cases').reset_index()
    weekday_totals = weekday_counts.groupby(['group', 'room', 'weekday']).cases.transform('sum')
    weekday_counts['share'] = weekday_counts.cases / weekday_totals
    weekday_blocks = []
    for (group, room, weekday), rows in weekday_counts.groupby(['group', 'room', 'weekday']):
        best = rows.sort_values(['share', 'service'], ascending=[False, True]).iloc[0]
        weekday_blocks.append({'group': group, 'room': room, 'weekday': int(weekday),
                               'dominant_service': str(best.service),
                               'dominant_share': float(best.share), 'cases': int(rows.cases.sum())})

    extreme = d[(d.actual < 10) | (d.actual > 720)][
        ['case_id', 'group', 'date', 'surgeon', 'service', 'booked', 'actual']].copy()
    extreme['date'] = extreme.date.dt.strftime('%Y-%m-%d')
    surgeon_days = d.groupby(['group', 'date', 'surgeon']).agg(
        cases=('case_id', 'size'), booked_minutes=('booked', 'sum'),
        actual_minutes=('actual', 'sum')).reset_index()
    extreme_surgeon_days = surgeon_days[surgeon_days.actual_minutes > 720].copy()
    extreme_surgeon_days['date'] = extreme_surgeon_days.date.dt.strftime('%Y-%m-%d')

    audit = {
        'flow': flow,
        'counts': d.groupby(['group', 'split']).size().rename('cases').reset_index().to_dict('records'),
        'closures': [str(x.date()) for x in closures], 'closure_training_median': median,
        'invalid_records_without_usable_date': unknown_dates,
        'booking_grid_verified': bool(booking.notna().all() and booking.mod(5).eq(4).all()),
        'multi_group_surgeon_dates': int(d.groupby(['surgeon', 'date']).group.nunique().gt(1).sum()),
        'multi_room_group_surgeon_dates': int(d.groupby(keys).room.nunique().gt(1).sum()),
        'time_window': '07:00 <= room entry < 17:00; otherwise DTT calendar date <= surgery date - 1 day',
        'invalid_date_key': 'First usable timestamp: room entry, surgical start, surgical stop, room exit',
        'cancellation_reasons': cancellation_reasons,
        'patient_type_counts': [{'patient_type': str(k) if str(k) else 'BLANK', 'cases': int(v)}
                                for k, v in d.patient_type.value_counts(dropna=False).items()],
        'patient_recurrence': {'unique_patients': int(patient_counts.size),
                               'patients_with_multiple_cases': int(repeat.size),
                               'cases_from_repeat_patients': int(repeat.sum()),
                               'patients_in_both_train_and_test': int(split_counts.gt(1).sum())},
        'same_day_extra_non_cancelled_cases': int(len(extras)),
        'same_day_extra_surgeon_dates': int(extras.groupby(['surgeon', 'date']).ngroups),
        'booking_distribution': [_booking_summary(d, 'All')] +
                                [_booking_summary(rows, group) for group, rows in d.groupby('group')],
        'specialty_room_day_concentration': block_summary,
        'specialty_room_weekday_blocks': weekday_blocks,
        'extreme_cases': extreme.to_dict('records'),
        'extreme_surgeon_days': extreme_surgeon_days.to_dict('records'),
    }
    return d, audit, pd.concat(excluded, ignore_index=True), quality


class Encoder:
    """Offline training effects; each training date is scored from all other training dates."""

    def __init__(self):
        self.shrinkage = {'procedure': None, 'surgeon': None}

    @staticmethod
    def _summary(frame):
        local = frame.copy()
        local['_error'] = local.actual - local.booked
        service = {}
        for key, rows in local.groupby('service', sort=False):
            values = rows._error.to_numpy(float)
            service[key] = (len(values), float(values.mean()), pairwise_spread(values))
        category = {}
        for name in ('procedure', 'surgeon'):
            cells = {}
            for key, rows in local.groupby(['service', name], sort=False):
                values = rows._error.to_numpy(float)
                cells[key] = (len(values), float(values.mean()), pairwise_spread(values))
            category[name] = cells
        return {'service': service, **category}

    @staticmethod
    def _shrinkage(frame, name):
        local = frame.copy()
        local['_error'] = local.actual - local.booked
        cells = {}
        for key, rows in local.groupby(['service', name], sort=False):
            values = rows._error.to_numpy(float)
            cells[key] = (len(values), float(values.sum()), float(values @ values))
        services = {}
        for key, rows in local.groupby('service', sort=False):
            values = rows._error.to_numpy(float)
            services[key] = (len(values), float(values.sum()), float(values @ values))
        n = sum(v[0] for v in cells.values())
        k, s = len(cells), len(services)
        if n <= k or k <= s:
            return None
        within = max(0., sum(q - total * total / count for count, total, q in cells.values())) / (n - k)
        between = (sum(total * total / count for count, total, _ in cells.values())
                   - sum(total * total / count for count, total, _ in services.values())) / (k - s)
        squared_counts = {service: 0 for service in services}
        for (service, _), (count, _, _) in cells.items():
            squared_counts[service] += count * count
        effective = sum(count - squared_counts[service] / count
                        for service, (count, _, _) in services.items()) / (k - s)
        variance = (between - within) / effective if effective > 0 else 0
        return within / variance if variance > 0 else None

    def _scores(self, frame, summary=None):
        summary = self.full_summary if summary is None else summary
        result = np.zeros((len(frame), 4))
        for i, row in enumerate(frame.itertuples()):
            ns, service_mean, service_spread = summary['service'].get(row.service, (0, 0., 0.))
            for j, name in enumerate(('procedure', 'surgeon')):
                n, mean, spread = summary[name].get((row.service, getattr(row, name)), (0, 0., 0.))
                k = self.shrinkage[name]
                if n and ns and k is not None:
                    weight = n / (n + k)
                    result[i, j] = weight * (mean - service_mean)
                    if n >= 2 and ns >= 2:
                        result[i, j + 2] = weight * (spread - service_spread)
        return result

    def _raw(self, frame, scores):
        return np.column_stack([frame.booked.to_numpy(float),
                                *[frame.service.eq(s).to_numpy(float) for s in self.services], scores])

    def fit_transform(self, train):
        local = train.reset_index(drop=True)
        counts = local.service.value_counts()
        self.reference = min(counts[counts == counts.max()].index)
        self.services = sorted(set(local.service) - {self.reference})
        self.shrinkage = {name: self._shrinkage(local, name) for name in ('procedure', 'surgeon')}
        self.full_summary = self._summary(local)
        self.training_scores = np.zeros((len(local), 4))
        for date, rows in local.groupby('date', sort=True):
            history = local[local.date.ne(date)]
            self.training_scores[rows.index] = self._scores(rows, self._summary(history))
        raw = self._raw(local, self.training_scores)
        self.mean, self.scale = raw.mean(axis=0), raw.std(axis=0)
        self.scale[self.scale < 1e-12] = 1
        self.names = ['intercept', 'booked', *[f'service:{s}' for s in self.services],
                      'procedure_bias', 'surgeon_bias', 'procedure_spread', 'surgeon_spread']
        return np.column_stack([np.ones(len(local)), (raw - self.mean) / self.scale])

    def transform(self, test):
        raw = self._raw(test, self._scores(test))
        return np.column_stack([np.ones(len(test)), (raw - self.mean) / self.scale])

    def metadata(self):
        return {'names': self.names, 'reference_service': self.reference,
                'mean': self.mean, 'scale': self.scale, 'shrinkage': self.shrinkage,
                'training_score_columns': ['procedure_bias', 'surgeon_bias',
                                           'procedure_spread', 'surgeon_spread'],
                'score_construction': 'training: all training dates except own date; test: all training data',
                'spread': 'average pairwise absolute error difference; singleton category spread score is zero',
                'spread_reliability': 'reuses the matching mean-effect shrinkage constant'}


@dataclass
class Day:
    key: str
    group: str
    date: str
    rooms: tuple
    case_ids: np.ndarray
    surgeon_ids: tuple
    case_surgeon: np.ndarray
    eligible: tuple
    booked: np.ndarray
    actual: np.ndarray
    historical: np.ndarray
    rows: np.ndarray
    weekday_fallback_surgeons: int = 0
    weekday_fallback_cases: int = 0
    fallback_surgeons: int = 0
    fallback_cases: int = 0
    eligibility_source: tuple = ()

    @property
    def n_surgeons(self):
        return len(self.surgeon_ids)

    @property
    def n_rooms(self):
        return len(self.rooms)

    def signature(self):
        return {'key': self.key, 'cases': self.case_ids, 'surgeons': self.case_surgeon,
                'surgeon_ids': self.surgeon_ids, 'rooms': self.rooms, 'eligible': self.eligible}


def build_days(frame, training):
    all_week = {(g, s): set(rows.room) for (g, s), rows in training.groupby(['group', 'service'])}
    weekday_training = training.assign(weekday=training.date.dt.weekday)
    by_weekday = {(g, int(w), s): set(rows.room)
                  for (g, w, s), rows in weekday_training.groupby(['group', 'weekday', 'service'])}
    days = []
    for (group, date), rows in frame.groupby(['group', 'date'], sort=True):
        rooms = tuple(sorted(rows.room.unique()))
        surgeons = tuple(sorted(rows.surgeon.unique()))
        eligible, source = [], []
        weekday_fallback_surgeons = weekday_fallback_cases = 0
        fallback_surgeons = fallback_cases = 0
        weekday = int(date.weekday())
        for surgeon in surgeons:
            cases = rows[rows.surgeon == surgeon]
            services = cases.service.unique()

            allowed = set(rooms)
            for service in services:
                allowed &= by_weekday.get((group, weekday, service), set())
            level = 'weekday'

            if not allowed:
                weekday_fallback_surgeons += 1
                weekday_fallback_cases += len(cases)
                allowed = set(rooms)
                for service in services:
                    allowed &= all_week.get((group, service), set())
                level = 'all_week_fallback'

            if not allowed:
                fallback_surgeons += 1
                fallback_cases += len(cases)
                allowed = set(rooms)
                level = 'candidate_room_fallback'

            eligible.append(tuple(i for i, room in enumerate(rooms) if room in allowed))
            source.append(level)

        days.append(Day(f'{group}_{date:%Y-%m-%d}', group, str(date.date()), rooms,
                        rows.case_id.to_numpy(), surgeons,
                        np.array([surgeons.index(s) for s in rows.surgeon]), tuple(eligible),
                        rows.booked.to_numpy(float), rows.actual.to_numpy(float),
                        np.array([rooms.index(r) for r in rows.room]), rows.index.to_numpy(),
                        weekday_fallback_surgeons, weekday_fallback_cases,
                        fallback_surgeons, fallback_cases, tuple(source)))
    return days
