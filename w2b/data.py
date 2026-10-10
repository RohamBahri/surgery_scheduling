"""Construct fixed-roster W2b weeks from the frozen daily cohort.

Historical surgeon operating days and case decision dates are explicit *availability
proxies*, not observed surgeon/patient availability. Roster/compatibility/weekday
frequencies are fitted on training dates only. No test outcomes enter fitting.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from surgery.data import COLUMNS, GROUPS, Encoder, build_cohort


@dataclass
class Week:
    group: str
    monday: str
    cases: np.ndarray
    booked: np.ndarray
    actual: np.ndarray
    X: np.ndarray
    surgeon: np.ndarray
    original_day: np.ndarray
    days_required: np.ndarray
    allowed_days: tuple
    slots: tuple                 # (weekday 0..4, named room)
    arcs: tuple                  # (case index, slot index)
    compatibility_fallbacks: int

    @property
    def n(self):
        return len(self.cases)

    @property
    def n_surgeons(self):
        return len(self.days_required)


def _weekday_history(train):
    usual = {}
    observed = train.assign(weekday=train.date.dt.weekday)
    for (group, surgeon), rows in observed.groupby(["group", "surgeon"]):
        # Count distinct operating dates, not cases.
        days = rows[["date", "weekday"]].drop_duplicates()
        counts = days.weekday.value_counts()
        usual[group, surgeon] = tuple(sorted(
            counts.index.tolist(), key=lambda d: (-int(counts[d]), int(d))
        )[:2])
    return usual


def _staffed_roster(train):
    """Training-only weekday room template; fixed before target-week scheduling."""
    out = {}
    temp = train.assign(weekday=train.date.dt.weekday)
    activity = temp[["group", "date", "weekday", "room"]].drop_duplicates()
    for group in GROUPS:
        subset = activity[activity.group.eq(group)]
        for weekday in range(5):
            rows = subset[subset.weekday.eq(weekday)]
            if rows.empty:
                raise ValueError(f"No training room activity for {group}, weekday {weekday}")
            sizes = rows.groupby("date").size()
            k = max(1, int(np.floor(float(sizes.median()) + 0.5)))
            counts = rows.room.value_counts()
            rooms = sorted(counts.index, key=lambda r: (-int(counts[r]), str(r)))[:k]
            out[group, weekday] = tuple(rooms)
    return out


def _compatibility(train):
    weekday = {}
    allweek = {}
    observed = train.assign(weekday=train.date.dt.weekday)
    for (group, day, service), rows in observed.groupby(["group", "weekday", "service"]):
        weekday[group, int(day), service] = frozenset(rows.room)
    for (group, service), rows in observed.groupby(["group", "service"]):
        allweek[group, service] = frozenset(rows.room)
    return weekday, allweek


def make_week(rows, X, roster, usual, compatibility, max_move_days=4):
    rows = rows.reset_index(drop=True)
    group = str(rows.group.iloc[0])
    monday = pd.Timestamp(rows.date.min()).normalize() - pd.Timedelta(
        days=int(pd.Timestamp(rows.date.min()).weekday()))
    surgeons = tuple(sorted(rows.surgeon.unique()))
    surgeon_index = {s: i for i, s in enumerate(surgeons)}
    original = rows.date.dt.weekday.to_numpy(int)
    surgeon = np.array([surgeon_index[s] for s in rows.surgeon], int)
    slots = tuple((d, r) for d in range(5) for r in roster[group, d])
    slot_lookup = {d: tuple(j for j, (day, _) in enumerate(slots) if day == d)
                   for d in range(5)}
    by_weekday, allweek = compatibility
    required = []
    allowed_days = []
    arcs = []
    fallbacks = 0
    dtt = pd.to_datetime(rows.dtt, errors="coerce")
    for s in surgeons:
        mask = rows.surgeon.eq(s)
        original_days = set(rows.loc[mask, "date"].dt.weekday.to_numpy(int))
        required.append(len(original_days))
        eligible_days = tuple(sorted(original_days | set(usual.get((group, s), ()))))
        allowed_days.append(eligible_days)
        # One surgeon/day has one room. Use an intersection of service eligibility;
        # fall back to any roster room when the intersection is empty.
        services = tuple(sorted(rows.loc[mask, "service"].unique()))
        for d in eligible_days:
            room_set = set(roster[group, d])
            for service in services:
                known = by_weekday.get((group, d, service), frozenset())
                if not known:
                    known = allweek.get((group, service), frozenset())
                room_set &= set(known)
            if not room_set:
                fallbacks += 1
                room_set = set(roster[group, d])
            for i in np.flatnonzero(mask.to_numpy()):
                # Original date always remains informationally eligible. Other
                # days require a recorded decision date no later than that day.
                if abs(int(original[i]) - d) > max_move_days:
                    continue
                candidate_date = monday + pd.Timedelta(days=d)
                if d != original[i] and (pd.isna(dtt.iloc[i]) or
                                          pd.Timestamp(dtt.iloc[i]).normalize() > candidate_date):
                    continue
                for j in slot_lookup[d]:
                    if slots[j][1] in room_set:
                        arcs.append((int(i), int(j)))
    by_case = {i: [] for i in range(len(rows))}
    for i, j in arcs:
        by_case[i].append(j)
    if any(not v for v in by_case.values()):
        raise ValueError(f"No W2b arcs for {group} {monday.date()}; check roster and decision dates")
    return Week(group, str(monday.date()), rows.case_id.to_numpy(),
                rows.booked.to_numpy(float), rows.actual.to_numpy(float),
                np.asarray(X, float), surgeon, original,
                np.asarray(required, int), tuple(allowed_days), slots,
                tuple(arcs), fallbacks)


def load_weeks(workbook: Path, *, group: str, split="train", weeks=2,
               max_cases=70, max_move_days=4):
    """Return training-fitted features and weeks. weeks=0 / max_cases=0 means all."""
    if group not in GROUPS:
        raise ValueError(f"Unknown group: {group}")
    raw = pd.read_excel(workbook, usecols=COLUMNS)
    cohort, audit, _, _ = build_cohort(raw)
    if not audit["booking_grid_verified"]:
        raise ValueError("Historical bookings are not on the verified five-minute-minus-one grid")
    train = cohort[cohort.split.eq("train")].copy()
    target = cohort[cohort.split.eq(split) & cohort.group.eq(group)].copy()
    if target.empty:
        raise ValueError("Empty requested cohort")
    group_train = train[train.group.eq(group)].reset_index(drop=True)
    encoder = Encoder()
    X_train = encoder.fit_transform(group_train)
    if split == "train":
        target = group_train.copy()
        X = X_train
    else:
        target = target.reset_index(drop=True)
        X = encoder.transform(target)
    target["monday"] = (target.date - pd.to_timedelta(target.date.dt.weekday, unit="D"))
    roster = _staffed_roster(train)
    usual = _weekday_history(train)
    compatible = _compatibility(train)
    chunks = []
    for monday, rows in target.groupby("monday", sort=True):
        if max_cases and len(rows) > max_cases:
            continue
        chunks.append((monday, rows))
    if weeks:
        chunks = chunks[:weeks]
    if not chunks:
        raise ValueError("No weeks under case limit; raise --max-cases or set it to 0")
    result = []
    for _, rows in chunks:
        result.append(make_week(rows, X[rows.index.to_numpy()], roster, usual,
                                compatible, max_move_days))
    meta = {"group": group, "split": split, "source_rows": len(raw),
            "retained_train": int(len(train)), "retained_test": int(
                cohort.split.eq("test").sum()), "group_cases": int(len(target)),
            "selected_weeks": len(result), "selected_cases": sum(w.n for w in result),
            "features": encoder.names, "roster": {
                str(d): list(roster[group, d]) for d in range(5)},
            "availability": "observed surgeon weekdays OR top-two training weekdays; "
                            "case decision date <= destination date except original",
            "staffing": "training-median active rooms per weekday; proxy, not observed roster"}
    return result, meta
