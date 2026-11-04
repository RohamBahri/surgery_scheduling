"""Prepare the one UHN workbook for the retrospective paper experiment.

This script is intentionally specific to the study dataset, not a general ETL framework.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from surgery.data import Encoder, build_cohort
from surgery.weekly_data import build_weeks, performed_events


def assemble_components(cohort, events, part):
    rows = cohort.loc[cohort["split"].eq(part)].reset_index(drop=True)
    positions = {int(case_id): i for i, case_id in enumerate(rows.case_id)}
    items = []
    for week in build_weeks(cohort, events, split=part):
        for index, component in enumerate(week.components()):
            case_positions = np.array(
                [positions[int(case_id)] for case_id in component.case_ids], dtype=int
            )
            key = f"{component.group}_{component.monday}_{index}"
            items.append((key, component, case_positions))
    used = np.concatenate([indices for _, _, indices in items])
    if len(used) != len(rows) or len(np.unique(used)) != len(rows):
        raise ValueError(f"{part} components do not partition the cases")
    return rows, items


def prepare(workbook: Path, output: Path, split_date: str = "2013-01-01"):
    if not workbook.is_file():
        raise FileNotFoundError(workbook)
    cutoff = pd.Timestamp(split_date)
    if not pd.Timestamp("2011-07-01") < cutoff < pd.Timestamp("2013-07-01"):
        raise ValueError("Split date must lie within the study period")
    raw = pd.read_excel(workbook)
    cohort, audit, excluded, quality = build_cohort(raw)
    cohort["split"] = np.where(cohort.date < cutoff, "train", "test")
    cohort = cohort.sort_values(["group", "date", "surgeon", "case_id"]).reset_index(drop=True)
    events = performed_events(raw)
    training, train_components = assemble_components(cohort, events, "train")
    evaluation, test_components = assemble_components(cohort, events, "test")
    if training.empty or evaluation.empty:
        raise ValueError("Both study periods must contain cases")

    encoder = Encoder()
    train_features = encoder.fit_transform(training)
    test_features = encoder.transform(evaluation)
    # Structural features consume surgeon and procedure spread, respectively.
    train_spread = encoder.training_scores[:, [3, 2]].copy()
    test_spread = encoder._scores(evaluation)[:, [3, 2]].copy()
    context = {
        "fit": dict(booked=training.booked.to_numpy(float),
                    actual=training.actual.to_numpy(float),
                    case_id=training.case_id.to_numpy(int), X=train_features),
        "held": dict(booked=evaluation.booked.to_numpy(float),
                     actual=evaluation.actual.to_numpy(float),
                     case_id=evaluation.case_id.to_numpy(int), X=test_features),
        "H": train_spread, "H_eval": test_spread,
        "train": train_components, "eval": test_components,
        "metadata": dict(split_date=str(cutoff.date()), workbook_sha256=hashlib.sha256(
            workbook.read_bytes()).hexdigest(), training_cases=len(training),
            evaluation_cases=len(evaluation), training_components=len(train_components),
            evaluation_components=len(test_components)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump(context, stream, protocol=pickle.HIGHEST_PROTOCOL)
    metadata = context["metadata"]
    audit["counts"] = cohort.groupby(["group", "split"]).size().rename(
        "cases").reset_index().to_dict("records")
    (output.parent / "cohort_audit.json").write_text(
        json.dumps({"metadata": metadata, "audit": audit}, indent=2, default=str))
    excluded.to_csv(output.parent / "excluded_cases.csv", index=False)
    quality.to_csv(output.parent / "data_quality.csv", index=False)
    print(json.dumps(metadata, indent=2))
    return context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/context.pkl"))
    parser.add_argument("--split-date", default="2013-01-01")
    args = parser.parse_args()
    prepare(args.workbook, args.out, args.split_date)


if __name__ == "__main__":
    main()
