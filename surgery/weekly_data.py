"""Weekly physical-room case loading; builds on the audited daily cohort.

Rosters and observed surgeon days are retrospective proxies, not known advance
staffing or surgeon-availability records. Historical support is strictly dated
before each planning week; only the explicitly marked excluded-case evidence
uses observations from that target week.
"""
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import GROUPS, ROOMS, identifier


@dataclass
class Week:
    group: str
    monday: str
    slots: tuple                  # (weekday 0..4, physical room)
    case_ids: np.ndarray
    surgeon_ids: tuple
    surgeon: np.ndarray          # local surgeon index for each case
    original: np.ndarray         # original physical room-day index for each case
    allowed: tuple               # allowed slot indices, one tuple per surgeon
    required_days: tuple         # original operating weekdays, per surgeon
    booked: np.ndarray
    actual: np.ndarray
    added_days: tuple            # (surgeon id, weekday, room, evidence), if any

    @property
    def n(self):
        return len(self.case_ids)

    @property
    def k(self):
        return len(self.slots)

    def validate(self):
        if not self.n or len(self.original) != self.n or len(self.surgeon) != self.n:
            raise ValueError("Invalid weekly case arrays")
        if len(self.allowed) != len(self.surgeon_ids) or len(self.required_days) != len(self.surgeon_ids):
            raise ValueError("Invalid weekly surgeon arrays")
        if len(self.booked) != self.n or len(self.actual) != self.n:
            raise ValueError("Duration count mismatch")
        if not np.isfinite(self.booked).all() or (self.booked <= 0).any():
            raise ValueError("Invalid booked durations")
        if not np.isfinite(self.actual).all() or (self.actual <= 0).any():
            raise ValueError("Invalid actual durations")
        for i, s in enumerate(self.surgeon):
            if int(self.original[i]) not in self.allowed[int(s)]:
                raise ValueError(f"Original session unavailable for case {i}")
        for s, days in enumerate(self.required_days):
            if not days or any(not any(self.slots[j][0] == d for j in self.allowed[s]) for d in days):
                raise ValueError("Original operating day cannot retain a case")
        return self

    def components(self):
        """Exactly decomposes by surgeons linked through eligible physical room-days."""
        parent = list(range(len(self.surgeon_ids)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        owner = {}
        for s, eligible in enumerate(self.allowed):
            for j in eligible:
                if j in owner:
                    parent[find(s)] = find(owner[j])
                else:
                    owner[j] = s
        groups = defaultdict(list)
        for s in range(len(parent)):
            groups[find(s)].append(s)
        result = []
        for members in groups.values():
            members = sorted(members)
            cases = [i for i, s in enumerate(self.surgeon) if int(s) in members]
            slots = sorted({j for s in members for j in self.allowed[s]})
            sl = {j: t for t, j in enumerate(slots)}
            su = {s: t for t, s in enumerate(members)}
            subset = Week(
                self.group, self.monday, tuple(self.slots[j] for j in slots),
                self.case_ids[cases].copy(),
                tuple(self.surgeon_ids[s] for s in members),
                np.asarray([su[int(self.surgeon[i])] for i in cases], dtype=int),
                np.asarray([sl[int(self.original[i])] for i in cases], dtype=int),
                tuple(tuple(sl[j] for j in self.allowed[s]) for s in members),
                tuple(self.required_days[s] for s in members),
                self.booked[cases].copy(), self.actual[cases].copy(),
                tuple(a for a in self.added_days if a[0] in set(self.surgeon_ids[s] for s in members))
            )
            result.append(subset.validate())
        return result


def performed_events(raw):
    """Noncancelled, nonemergency surgeries with valid performed-room timestamps.

    Includes otherwise excluded cases, for availability evidence only. Never
    uses actual *durations* to choose eligibility.
    """
    room = raw.Operating_Room.fillna("").astype(str).str.upper().str.replace(r"\s+", "", regex=True)
    lookup = {r: group for group, rooms in ROOMS.items() for r in rooms}
    group = room.map(lookup)
    surgeon = identifier(raw.Surgeon_Code)
    enter = (pd.to_datetime(raw["Enter Room Date"], errors="coerce").dt.normalize()
             + pd.to_timedelta(raw["Enter Room Time"], errors="coerce"))
    start = (pd.to_datetime(raw["Actual Start Date"], errors="coerce").dt.normalize()
             + pd.to_timedelta(raw["Actual Start Time"], errors="coerce"))
    stop = (pd.to_datetime(raw["Actual Stop Date"], errors="coerce").dt.normalize()
            + pd.to_timedelta(raw["Actual Stop Time"], errors="coerce"))
    leave = (pd.to_datetime(raw["Leave Room Date"], errors="coerce").dt.normalize()
             + pd.to_timedelta(raw["Leave Room Time"], errors="coerce"))
    date = enter.dt.normalize()
    cancelled = (raw.Case_Cancelled_Reason.fillna("").astype(str).str.strip().ne("")
                 | raw["Case Cancel Date"].notna())
    emergency = raw.Patient_Type.fillna("").astype(str).str.upper().str.contains("EMERGENCY")
    valid = (group.notna() & surgeon.ne("") & ~cancelled & ~emergency
             & enter.le(start) & start.lt(stop) & stop.le(leave)
             & (leave - enter).dt.total_seconds().div(60).between(0, 1440, inclusive="neither")
             & date.dt.weekday.lt(5) & date.between("2011-07-01", "2013-06-30"))
    frame = pd.DataFrame({
        "case_id": np.arange(len(raw)) + 2, "group": group, "surgeon": surgeon,
        "room": room, "date": date
    }).loc[valid].copy()
    frame["weekday"] = frame.date.dt.weekday.astype(int)
    frame["monday"] = frame.date - pd.to_timedelta(frame.weekday, unit="D")
    return frame.sort_values(["group", "surgeon", "date", "room", "case_id"]).reset_index(drop=True)


def build_weeks(cohort, events, *, split=None, group=None, prior_dates=1):
    """Construct fixed-roster, surgeon-eligible weeks without changing the cohort.

    An extra weekday is considered only for a surgeon observed on one weekday
    with >=2 retained cases. Candidate room must be active in the target week.
    Support: >=prior_dates distinct earlier performed dates on the candidate
    weekday/room, OR a performed case excluded by cleaning in that same week.
    Exactly one candidate (weekday, room) is selected deterministically.
    """
    if prior_dates < 1:
        raise ValueError("prior_dates must be positive")
    frame = cohort.copy()
    if split is not None:
        frame = frame[frame.split.eq(split)].copy()
    if group is not None:
        frame = frame[frame.group.eq(group)].copy()
    if frame.empty:
        return []
    frame["weekday"] = frame.date.dt.weekday.astype(int)
    frame["monday"] = frame.date - pd.to_timedelta(frame.weekday, unit="D")
    retained = set(cohort.case_id)
    removed = events[~events.case_id.isin(retained)]
    excluded = defaultdict(list)
    for ev in removed.itertuples(index=False):
        excluded[(ev.group, ev.surgeon, ev.monday)].append((int(ev.weekday), str(ev.room)))
    previous = defaultdict(list)
    for ev in events.itertuples(index=False):
        previous[(ev.group, ev.surgeon, int(ev.weekday))].append((ev.date, str(ev.room)))

    result = []
    for (g, monday), rows in frame.groupby(["group", "monday"], sort=True):
        rows = rows.sort_values("case_id").reset_index(drop=True)
        slots = tuple(sorted(set(zip(rows.weekday.astype(int), rows.room))))
        lookup = {key: j for j, key in enumerate(slots)}
        surgeons = tuple(sorted(rows.surgeon.unique()))
        surgeon_idx = {s: j for j, s in enumerate(surgeons)}
        case_surgeon = np.asarray([surgeon_idx[s] for s in rows.surgeon], dtype=int)
        original = np.asarray([lookup[int(wd), room] for wd, room
                               in zip(rows.weekday, rows.room)], dtype=int)
        allowed, required, added = [], [], []
        by_weekday = defaultdict(list)
        for j, (weekday, room) in enumerate(slots):
            by_weekday[weekday].append((room, j))
        for s in surgeons:
            positions = np.flatnonzero(rows.surgeon.eq(s).to_numpy())
            primary = sorted(set(int(original[i]) for i in positions))
            days = tuple(sorted({slots[j][0] for j in primary}))
            eligible = set(primary)
            if len(days) == 1 and len(positions) >= 2:
                alternatives = []
                raw_same_week = excluded.get((g, s, monday), ())
                for wd, candidates in sorted(by_weekday.items()):
                    if wd in days:
                        continue
                    prior = [(dt, room) for dt, room in previous.get((g, s, wd), ()) if dt < monday]
                    same_week_rooms = {room for day, room in raw_same_week if day == wd}
                    for room, j in candidates:
                        dates = {dt for dt, r in prior if r == room}
                        direct = room in same_week_rooms
                        if len(dates) < prior_dates and not direct:
                            continue
                        # Direct recovered operating-day evidence takes priority.
                        alternatives.append((
                            -int(direct), -len(dates),
                            -sum(r == room for _, r in prior), wd, room, j,
                            "excluded_performed" if direct else "prior_dates"
                        ))
                if alternatives:
                    chosen = min(alternatives)
                    _, _, _, wd, room, j, source = chosen
                    eligible.add(j)
                    added.append((s, int(wd), room, source))
            allowed.append(tuple(sorted(eligible)))
            required.append(days)
        week = Week(
            str(g), str(pd.Timestamp(monday).date()), slots,
            rows.case_id.to_numpy(dtype=int), surgeons, case_surgeon, original,
            tuple(allowed), tuple(required),
            rows.booked.to_numpy(float), rows.actual.to_numpy(float),
            tuple(added)
        )
        result.append(week.validate())
    return result
