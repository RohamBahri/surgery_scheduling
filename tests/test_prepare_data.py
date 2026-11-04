"""Synthetic workbook test of the UHN-only cohort preparation stage."""
import pandas as pd

from pilot.prepare_data import prepare


def test_prepare_from_single_workbook(tmp_path):
    rows = []
    for date, patient in [("2011-08-01", "P1"), ("2013-02-04", "P2")]:
        row = {
            "Operating_Room": "OR1", "Surgeon_Code": "S1",
            "Main_Procedure_Id": "PROC1", "Case_Service": "SURG",
            "Booked Time (Minutes)": 119, "Decision_Date": pd.Timestamp(date) -
                pd.Timedelta(days=2), "Patient_Type": "ELECTIVE",
            "Patient_ID": patient, "Case_Cancelled_Reason": "",
            "Case Cancel Date": pd.NaT,
        }
        for label, minute in [
            ("Enter Room", "08:00:00"), ("Actual Start", "08:10:00"),
            ("Actual Stop", "09:00:00"), ("Leave Room", "09:10:00")]:
            row[label + " Date"] = pd.Timestamp(date)
            row[label + " Time"] = minute
        rows.append(row)
    file = tmp_path / "UHN.xlsx"
    pd.DataFrame(rows).to_excel(file, index=False)
    checkpoint = tmp_path / "prepared" / "context.pkl"
    study = prepare(file, checkpoint)
    assert checkpoint.is_file()
    assert len(study["fit"]["booked"]) == 1
    assert len(study["held"]["booked"]) == 1
    assert study["fit"]["X"].shape[1] == study["held"]["X"].shape[1]
    assert study["H"].shape == study["H_eval"].shape == (1, 2)
    assert len(study["train"]) == len(study["eval"]) == 1
    assert (checkpoint.parent / "retained_cases.csv").is_file()
