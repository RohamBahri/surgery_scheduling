"""Rich, low-dimensional CASE-LEVEL correction features for the UHN OR project.

Drop-in wrapper around the EXISTING surgery.data.Encoder: its leave-one-date-out
training scores, shrinkage, and test-time training-history logic are untouched.
The standalone booked-duration column is REMOVED from the policy design matrix.
It remains the exact unit-coefficient anchor OUTSIDE the correction:

    advice_i = X_i @ w
    recommended_duration_i = booked_i + advice_i
    post_review_duration_i = booked_i + response(display(advice_i), alpha, h)

The design includes bounded booked-duration *interactions*, not an unrestricted
booked-duration main effect. This preserves the explicit unit anchor, although
allowed interactions can alter effective case-specific slopes.

No post-surgery fields are used as contemporaneous predictors. Historical
outcome-derived features use the same leave-one-date-out convention for
training and training-only aggregates at transform time.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from surgery.data import Encoder

REQUIRED = ('case_id', 'date', 'group', 'service', 'procedure',
            'surgeon', 'booked', 'patient_type')
AGG_COLS = ('n', 'err_sum', 'abs_sum', 'under_sum')


def _date_series(values: pd.Series) -> pd.Series:
    """Read pandas Excel datetimes or genuine Excel serial dates, not Unix ns."""
    values = pd.Series(values).reset_index(drop=True)
    numerical = pd.to_numeric(values, errors='coerce')
    is_serial = numerical.between(20000, 80000)
    result = pd.to_datetime(values.where(~is_serial), errors='coerce')
    if is_serial.any():
        result.loc[is_serial] = (pd.Timestamp('1899-12-30') +
                                 pd.to_timedelta(numerical.loc[is_serial], unit='D'))
    return result.dt.normalize()


def attach_preoperative_dates(cohort: pd.DataFrame, workbook_raw: pd.DataFrame) -> pd.DataFrame:
    """Attach Consult_Date without changing cohort rows/order or the existing Encoder.

    build_cohort assigns original workbook row IDs: row 2 -> case_id 2.
    Pass the original *unreordered* pd.read_excel(...) frame as workbook_raw.
    Add Consult_Date to read_excel(usecols=...) at load time. If unavailable,
    missing lead-time features are emitted consistently.
    """
    out = cohort.copy()
    if 'consult' in out.columns:
        out['consult'] = _date_series(out['consult']).to_numpy()
        return out
    if 'Consult_Date' not in workbook_raw.columns:
        out['consult'] = pd.NaT
        return out
    ids = out.case_id.to_numpy(dtype=np.int64) - 2
    if (ids < 0).any() or (ids >= len(workbook_raw)).any():
        raise ValueError('case_id does not correspond to a row of workbook_raw')
    raw_dates = _date_series(workbook_raw['Consult_Date'])
    out['consult'] = raw_dates.iloc[ids].to_numpy()
    return out


def _enforce(frame: pd.DataFrame):
    missing = set(REQUIRED) - set(frame)
    if missing:
        raise ValueError(f'Missing required cohort fields: {sorted(missing)}')
    if frame.case_id.duplicated().any():
        raise ValueError('Duplicate case_id in feature frame')
    if not np.isfinite(pd.to_numeric(frame.booked, errors='coerce')).all() or (frame.booked <= 0).any():
        raise ValueError('Invalid bookings')
    if pd.to_datetime(frame.date, errors='coerce').isna().any():
        raise ValueError('Every row must have a known planned surgery date')


def _counts(frame: pd.DataFrame, keys: tuple[str, ...]) -> pd.DataFrame:
    a = frame.assign(
        _error=np.asarray(frame.actual, float) - np.asarray(frame.booked, float),
    )
    a['_abs'] = a._error.abs()
    a['_under'] = (a._error > 0).astype(float)
    grouped = a.groupby(list(keys), dropna=False, observed=True)
    result = grouped.agg(n=('_error', 'size'), err_sum=('_error', 'sum'),
                         abs_sum=('_abs', 'sum'), under_sum=('_under', 'sum'))
    return result.reset_index()


def _lookup(frame: pd.DataFrame, full: pd.DataFrame, keys: tuple[str, ...],
            *, dated: pd.DataFrame | None = None) -> pd.DataFrame:
    """Map aggregate totals to cases; for training subtract the OWN DATE exactly."""
    f = frame.reset_index(drop=True)
    merge_keys = list(keys)
    a = f[merge_keys].merge(full, on=merge_keys, how='left', sort=False)
    a[list(AGG_COLS)] = a[list(AGG_COLS)].fillna(0.)
    if dated is not None:
        delta = f[merge_keys + ['date']].merge(
            dated, on=merge_keys + ['date'], how='left', sort=False
        )
        a[list(AGG_COLS)] = a[list(AGG_COLS)].to_numpy(float) - delta[list(AGG_COLS)].fillna(0.).to_numpy(float)
    # Residual sums may legitimately be negative (systematic overbooking).
    # Only counts and nonnegative moments can be clipped for roundoff.
    for col in ('n', 'abs_sum', 'under_sum'):
        a[col] = a[col].clip(lower=0)
    return a[list(AGG_COLS)].reset_index(drop=True)


def _rate(summary: pd.DataFrame, numerator: str, prior: np.ndarray, pseudo: float) -> np.ndarray:
    n = summary.n.to_numpy(float)
    s = summary[numerator].to_numpy(float)
    return (s + pseudo * prior) / np.maximum(n + pseudo, 1.)


def _mean(summary: pd.DataFrame, numerator: str, fallback: np.ndarray) -> np.ndarray:
    n = summary.n.to_numpy(float)
    return np.divide(summary[numerator].to_numpy(float), n,
                     out=np.asarray(fallback, float).copy(), where=n > 0)


def _median_lead(frame: pd.DataFrame, column: str, max_days: float):
    if column not in frame:
        return None
    start = pd.to_datetime(frame.date, errors='coerce')
    end = _date_series(frame[column])
    lead = (start.reset_index(drop=True) - end).dt.days.astype(float)
    valid = lead.between(0, 10000).to_numpy()
    log = np.log1p(np.clip(lead.to_numpy(float), 0, max_days))
    return valid, log


@dataclass
class FeatureMetadata:
    names: list[str]
    n_features: int
    removed_base_feature: str
    train_cases: int
    same_date_exclusion: bool
    unit_booking_anchor: bool
    reference_service: str
    history_pseudocount: int
    scaled_extra_features: int


class RichCorrectionEncoder:
    """A fixed-size, regularized-linear-policy feature builder.

    Provides 60–90 columns for a typical UHN training cohort rather than
    thousands of procedure/surgeon indicators. Use ridge/elastic-net for the
    prediction loss or the SAME X for decision-aware optimization.

    Preserves Encoder.fit_transform() and Encoder.transform() exactly.
    """

    KEYS = {
        'global': (),
        'service': ('service',),
        'proc': ('service', 'procedure'),
        'surgeon': ('service', 'surgeon'),
        'pair': ('service', 'procedure', 'surgeon'),
    }

    def __init__(self, pseudo: int = 20, log_lead_clip: int = 730):
        if pseudo <= 0:
            raise ValueError('Pseudo-count must be positive')
        self.pseudo = int(pseudo)
        self.log_lead_clip = int(log_lead_clip)

    @staticmethod
    def _prepare(frame: pd.DataFrame):
        f = frame.reset_index(drop=True).copy()
        f['date'] = pd.to_datetime(f.date, errors='raise').dt.normalize()
        f['service'] = f.service.fillna('').astype(str)
        f['procedure'] = f.procedure.fillna('').astype(str)
        f['surgeon'] = f.surgeon.fillna('').astype(str)
        f['group'] = f.group.fillna('').astype(str)
        f['patient_type'] = f.patient_type.fillna('MISSING').astype(str).replace('', 'MISSING')
        for name in ('dtt', 'consult'):
            if name in f:
                f[name] = _date_series(f[name]).to_numpy()
            else:
                f[name] = pd.NaT
        _enforce(f)
        return f

    def _summaries(self, train: pd.DataFrame):
        self.full = {}
        self.same_date = {}
        for label, keys in self.KEYS.items():
            if not keys:
                # Global prior for the fold: total minus its own date.
                self.full[label] = _counts(train.assign(_everything='ALL'), ('_everything',))
                self.same_date[label] = _counts(train.assign(_everything='ALL'), ('_everything','date'))
            else:
                self.full[label] = _counts(train, keys)
                self.same_date[label] = _counts(train, keys + ('date',))

    def _history(self, frame: pd.DataFrame, *, train_mode: bool) -> dict[str, np.ndarray]:
        f = frame.copy()
        f['_everything'] = 'ALL'
        history = {}
        for label, keys in self.KEYS.items():
            actual_keys = keys or ('_everything',)
            history[label] = _lookup(f, self.full[label], actual_keys,
                                     dated=self.same_date[label] if train_mode else None)
        g, service, proc, surgeon, pair = (history[k] for k in ('global','service','proc','surgeon','pair'))
        # The global mean/prior also excludes the own date for training.
        gmean = _mean(g, 'err_sum', np.zeros(len(f)))
        gabs = _mean(g, 'abs_sum', np.zeros(len(f)))
        gunder = _mean(g, 'under_sum', np.zeros(len(f)))
        svmean = _rate(service, 'err_sum', gmean, self.pseudo)
        svabs = _rate(service, 'abs_sum', gabs, self.pseudo)
        svunder = _rate(service, 'under_sum', gunder, self.pseudo)
        pmean = _rate(proc, 'err_sum', svmean, self.pseudo)
        smean = _rate(surgeon, 'err_sum', svmean, self.pseudo)
        pabs = _rate(proc, 'abs_sum', svabs, self.pseudo)
        sabs = _rate(surgeon, 'abs_sum', svabs, self.pseudo)
        punder = _rate(proc, 'under_sum', svunder, self.pseudo)
        sunder = _rate(surgeon, 'under_sum', svunder, self.pseudo)
        pair_mean = _mean(pair, 'err_sum', .5 * (pmean+smean))
        pair_n = pair.n.to_numpy(float)
        pair_extra = pair_n / (pair_n + self.pseudo) * (pair_mean - .5 * (pmean+smean))
        return {
            'hist_service_mean_error': svmean,
            'hist_service_mean_abs_error': svabs,
            'hist_service_under_rate': svunder,
            'hist_procedure_mean_abs_error': pabs,
            'hist_surgeon_mean_abs_error': sabs,
            'hist_procedure_under_rate': punder,
            'hist_surgeon_under_rate': sunder,
            'hist_procedure_log_support': np.log1p(proc.n.to_numpy(float)),
            'hist_surgeon_log_support': np.log1p(surgeon.n.to_numpy(float)),
            'hist_joint_log_support': np.log1p(pair_n),
            'hist_pair_residual_interaction': pair_extra,
        }

    def _context(self, frame: pd.DataFrame, legacy: np.ndarray, train_mode: bool):
        f = frame
        out = {}
        for group in self.other_groups:
            out['site:'+group] = f.group.eq(group).to_numpy(float)
        for val in self.other_patient_types:
            out['patient_type:'+val] = f.patient_type.eq(val).to_numpy(float)
        day = f.date.dt.dayofyear.to_numpy(float)
        out['calendar_sin'] = np.sin(2*np.pi*day/365.25)
        out['calendar_cos'] = np.cos(2*np.pi*day/365.25)
        out['calendar_year_progress'] = (f.date-self.date_origin).dt.days.to_numpy(float) / 365.25
        for name in ('dtt', 'consult'):
            lead = _median_lead(f, name, self.log_lead_clip)
            valid, log = lead if lead is not None else (np.zeros(len(f), bool), np.zeros(len(f)))
            median = self.lead_medians.get(name, 0.)
            out[name+'_log_lead'] = np.where(valid,log,median)
            out[name+'_missing'] = (~valid).astype(float)
        out.update(self._history(f, train_mode=train_mode))
        # The unit booked anchor remains outside X. These are ONLY contextual
        # adjustments and can change the conditional slope in non-reference groups.
        relative_b = np.clip((f.booked.to_numpy(float)-180.)/180., -1., 5.)
        for group in self.other_groups:
            out['interaction_booked_x_site:'+group] = relative_b*f.group.eq(group).to_numpy(float)
        for service in self.base.services:
            out['interaction_booked_x_service:'+service] = relative_b*f.service.eq(service).to_numpy(float)
        out['interaction_booked_x_procedure_bias'] = relative_b*legacy[:,0]
        out['interaction_booked_x_surgeon_bias'] = relative_b*legacy[:,1]
        return out

    def fit_transform(self, train: pd.DataFrame) -> np.ndarray:
        f = self._prepare(train)
        if 'actual' not in f or not np.isfinite(f.actual.to_numpy(float)).all():
            raise ValueError('Training needs finite realized durations to estimate history')
        self.base = Encoder()
        old = self.base.fit_transform(f)  # IMPORTANT: unchanged LODO estimator
        self.original_names = list(self.base.names)
        self.keep = [i for i,s in enumerate(self.original_names) if s != 'booked']
        if len(self.keep) != len(self.original_names)-1:
            raise AssertionError('Expected exactly one free booked column in existing Encoder')
        self.reference_site = f.group.value_counts().index[0]
        self.other_groups = sorted(set(f.group)-{self.reference_site})
        self.reference_patient_type = f.patient_type.value_counts().index[0]
        self.other_patient_types = sorted(set(f.patient_type)-{self.reference_patient_type})
        self.date_origin = f.date.min()
        self.lead_medians = {}
        for name in ('dtt','consult'):
            valid, lead = _median_lead(f,name,self.log_lead_clip)
            self.lead_medians[name] = float(np.median(lead[valid])) if valid.any() else 0.
        self._summaries(f)
        extra = self._context(f,self.base.training_scores,train_mode=True)
        self.extra_names = list(extra)
        mat = np.column_stack([extra[k] for k in self.extra_names]).astype(float)
        self.extra_mean = np.mean(mat,axis=0)
        self.extra_std = np.std(mat,axis=0)
        self.extra_keep = self.extra_std > 1e-10
        self.extra_std[~self.extra_keep] = 1.
        self.names = [self.original_names[i] for i in self.keep] + [n for n,keep in zip(self.extra_names,self.extra_keep) if keep]
        self.train_cases = len(f)
        return self._combine(old,mat)

    def _combine(self, base_X: np.ndarray, extras: np.ndarray) -> np.ndarray:
        extra = ((extras-self.extra_mean)/self.extra_std)[:,self.extra_keep]
        X = np.column_stack([base_X[:,self.keep],extra])
        if not np.isfinite(X).all():
            raise ValueError('Non-finite generated features')
        return X

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not hasattr(self,'base'):
            raise RuntimeError('fit_transform must be called before transform')
        f = self._prepare(frame)
        old = self.base.transform(f)
        # The existing Encoder._scores uses the COMPLETE fitted TRAIN HISTORY.
        legacy = self.base._scores(f)
        extra = self._context(f,legacy,train_mode=False)
        mat = np.column_stack([extra[k] for k in self.extra_names]).astype(float)
        return self._combine(old,mat)

    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            names=list(self.names), n_features=len(self.names),
            removed_base_feature='booked',train_cases=self.train_cases,
            same_date_exclusion=True,unit_booking_anchor=True,
            reference_service=str(self.base.reference),
            history_pseudocount=self.pseudo,
            scaled_extra_features=int(self.extra_keep.sum()),
        )
