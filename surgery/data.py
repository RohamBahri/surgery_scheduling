"""UHN cohort, chronological features, and room-group/day instances."""
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
           'Booked Time (Minutes)', 'Decision_Date', 'Patient_Type',
           'Case_Cancelled_Reason', 'Case Cancel Date'] + [
               stem + suffix for stem in TIMESTAMPS for suffix in (' Date', ' Time')]


def identifier(series):
    return series.fillna('').astype(str).str.strip().str.replace(r'\.0$', '', regex=True)


def overlap_flags(frame):
    flagged = pd.Series(False, index=frame.index)
    for _, rows in frame.groupby(['room', 'date'], sort=False):
        start, end = rows.rin.to_numpy(), rows.rout.to_numpy()
        overlap = np.minimum(end[:, None], end) - np.maximum(start[:, None], start)
        bad = (overlap > np.timedelta64(15, 'm')) & ~np.eye(len(rows), dtype=bool)
        flagged.loc[rows.index] = bad.any(axis=1)
    return flagged


def build_cohort(raw):
    missing = set(COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f'Missing workbook columns: {sorted(missing)}')
    d = pd.DataFrame(index=raw.index)
    d['case_id'] = np.arange(len(raw)) + 2
    d['room'] = raw.Operating_Room.fillna('').str.upper().str.replace(r'\s+', '', regex=True)
    d['group'] = d.room.map({r: g for g, rooms in ROOMS.items() for r in rooms})
    for source, target in [('Surgeon_Code', 'surgeon'), ('Main_Procedure_Id', 'procedure'),
                           ('Case_Service', 'service')]:
        d[target] = identifier(raw[source])
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
    cancelled = (raw.Case_Cancelled_Reason.fillna('').astype(str).str.strip().ne('')
                 | raw['Case Cancel Date'].notna())
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
    audit = {
        'flow': flow, 'counts': d.groupby(['group', 'split']).size().rename('cases').reset_index().to_dict('records'),
        'closures': [str(x.date()) for x in closures], 'closure_training_median': median,
        'invalid_records_without_usable_date': unknown_dates,
        'booking_grid_verified': bool(booking.notna().all() and booking.mod(5).eq(4).all()),
        'multi_group_surgeon_dates': int(d.groupby(['surgeon', 'date']).group.nunique().gt(1).sum()),
        'multi_room_group_surgeon_dates': int(d.groupby(keys).room.nunique().gt(1).sum()),
        'time_window': '07:00 <= room entry < 17:00; otherwise DTT calendar date <= surgery date - 1 day',
        'invalid_date_key': 'First usable timestamp: room entry, surgical start, surgical stop, room exit',
    }
    return d, audit, pd.concat(excluded, ignore_index=True), quality


class Encoder:
    """Service/category error moments are updated only after a whole date is encoded."""

    def __init__(self):
        self.history = {'service': {}, 'procedure': {}, 'surgeon': {}}
        self.shrinkage = {'procedure': None, 'surgeon': None}

    def _shrinkage(self, name):
        cells = self.history[name]
        services = self.history['service']
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

    def _scores(self, frame):
        result = np.zeros((len(frame), 2))
        for i, row in enumerate(frame.itertuples()):
            ns, total, _ = self.history['service'].get(row.service, (0, 0., 0.))
            for j, name in enumerate(['procedure', 'surgeon']):
                n, error, _ = self.history[name].get((row.service, getattr(row, name)), (0, 0., 0.))
                k = self.shrinkage[name]
                if n and ns and k is not None:
                    result[i, j] = (error - n * total / ns) / (n + k)
        return result

    def _update(self, frame):
        for row in frame.itertuples():
            error = row.actual - row.booked
            for name in self.history:
                key = row.service if name == 'service' else (row.service, getattr(row, name))
                n, total, q = self.history[name].get(key, (0, 0., 0.))
                self.history[name][key] = n + 1, total + error, q + error * error
        self.shrinkage = {name: self._shrinkage(name) for name in ['procedure', 'surgeon']}

    def _raw(self, frame, scores):
        return np.column_stack([frame.booked.to_numpy(float),
                                *[frame.service.eq(s).to_numpy(float) for s in self.services], scores])

    def fit_transform(self, train):
        self.__init__()
        counts = train.service.value_counts()
        self.reference = min(counts[counts == counts.max()].index)
        self.services = sorted(set(train.service) - {self.reference})
        local = train.reset_index(drop=True)
        self.training_scores = np.zeros((len(local), 2))
        for _, rows in local.groupby('date', sort=True):
            self.training_scores[rows.index] = self._scores(rows)
            self._update(rows)
        raw = self._raw(local, self.training_scores)
        self.mean, self.scale = raw.mean(axis=0), raw.std(axis=0)
        self.scale[self.scale < 1e-12] = 1
        self.names = ['intercept', 'booked', *[f'service:{s}' for s in self.services],
                      'procedure_score', 'surgeon_score']
        return np.column_stack([np.ones(len(local)), (raw - self.mean) / self.scale])

    def transform(self, test):
        raw = self._raw(test, self._scores(test))
        return np.column_stack([np.ones(len(test)), (raw - self.mean) / self.scale])

    def metadata(self):
        return {'names': self.names, 'reference_service': self.reference,
                'mean': self.mean, 'scale': self.scale, 'final_shrinkage': self.shrinkage,
                'score_cutoff': 'strictly earlier dates, including shrinkage estimation'}


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
    fallback_surgeons: int = 0
    fallback_cases: int = 0

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
    history = {(g, s): set(rows.room) for (g, s), rows in training.groupby(['group', 'service'])}
    days = []
    for (group, date), rows in frame.groupby(['group', 'date'], sort=True):
        rooms = tuple(sorted(rows.room.unique()))
        surgeons = tuple(sorted(rows.surgeon.unique()))
        eligible, nf, nc = [], 0, 0
        for surgeon in surgeons:
            cases = rows[rows.surgeon == surgeon]
            allowed = set(rooms)
            for service in cases.service.unique():
                allowed &= history.get((group, service), set())
            if not allowed:
                allowed, nf, nc = set(rooms), nf + 1, nc + len(cases)
            eligible.append(tuple(i for i, room in enumerate(rooms) if room in allowed))
        days.append(Day(f'{group}_{date:%Y-%m-%d}', group, str(date.date()), rooms,
                        rows.case_id.to_numpy(), surgeons,
                        np.array([surgeons.index(s) for s in rows.surgeon]), tuple(eligible),
                        rows.booked.to_numpy(float), rows.actual.to_numpy(float),
                        np.array([rooms.index(r) for r in rows.room]), rows.index.to_numpy(), nf, nc))
    return days
