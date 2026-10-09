# UHN daily planning experiment

This repository implements one experiment on the UHN July 2011–June 2013 workbook. It uses TGH, TWH main, and TWH day surgery. Cases keep their dates and room groups; rooms may be released. PMH is excluded.

The five methods are Booked, Shift, Case-Error, VF-Direct, and VF. The four response scenarios are \((alpha,h)=(0.5,30),(0.5,60),(0.8,30),(0.8,60)\). VF-Direct is fitted once per group and evaluated through each response.

## Install and run

The exact package versions used by CI are in \`requirements-lock.txt\`. A full Gurobi license is required for the complete experiment.

\`\`\`bash
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m pytest -q

python run_experiment.py /path/to/UHNOperating_RoomScheduling2011-2013.xlsx --stage audit
python run_experiment.py /path/to/UHNOperating_RoomScheduling2011-2013.xlsx --stage pilot
\`\`\`

Then use \`--stage checks\`, \`baselines\`, \`pilot-vf\`, \`train\`, and finally \`evaluate\`. Repeating the same command resumes completed checkpoints. The run identity includes the workbook hash, scientific source code, scenarios, tie rule, seed, and numerical environment.

## Fixed design

- Training: July 2011–December 2012. Test: January–June 2013.
- One planning instance is one room group on one date; surgery dates never move.
- A surgeon's cases within a group/date stay in one room.
- Candidate rooms are rooms represented by the retained workload that day; rooms may be released.
- Room eligibility first uses each service's training rooms on the same weekday, then its rooms over all training weekdays, then all candidate rooms only if necessary. Both fallback levels are audited.
- Sessions are 480 minutes with 30 minutes between cases.
- Cost is one unit per idle minute and 1.75 per overtime minute.
- Production planning uses one exact Gurobi set-partitioning model over feasible room patterns. It minimizes cost, then the largest planned room load, then the alphabetical surgeon-to-room assignment. The compact Gurobi model remains only as an independent validation formulation in tests.
- Training features are booking, service indicators, and four offline history scores: procedure bias, surgeon bias, procedure spread, and surgeon spread. Each training date is scored from all other training dates; test cases use the complete training sample. Spread is the average pairwise absolute difference in booking error.
- Results remain separate across the four behavioral scenarios.

## Resume and exactness

Completed daily solves, accepted policy updates, Shift trials, growing VF libraries, oracle cuts, and feasible oracle bounds are checkpointed in SQLite. Each day has its own database. An unfinished ordinary planner remains pending and is retried; it is never treated as exact or dropped from reporting. An unfinished response-limited oracle may remain as a valid lower/upper bracket.

Independent days run in separate processes with \`--workers\`; \`--threads\` is the Gurobi thread count per worker. Case-Error fits run in blocks of 30 updates and resume if capped. Test evaluation requires converged Case-Error fits. P3's first VF iteration is reused by the full fit.

## Outputs

- \`audit/\`: cohort flow, retained/excluded Excel rows, cancellation and booking diagnostics, specialty-block summaries, weekday eligibility and fallback diagnostics, encoder metadata, and feature arrays.
- \`pilots_P1_P2.json\`, \`pilot_P3.json\`: training-only timing and certification records.
- \`checks/\`: Historical, Historical-one-room, Booked, and realized-duration planning checks.
- \`models_*.json\`, \`penalty_diagnostics.csv\`, \`training_diagnostics.json\`, \`training_certificates.csv\`: coefficients, fitting states, penalty diagnostics, VF trajectories, and training certificates.
- \`frozen_test_policies.json\`: the weights and shifts frozen before test evaluation.
- \`report/\`: daily results, case-level recommendations and outcomes, occupied room-day loads, method summaries, oracle brackets, opportunity ratios, and paired weekly bootstrap comparisons.
- \`checkpoints.sqlite\`, \`days/*/checkpoints.sqlite\`, \`solver_log.jsonl\`: resumable state and optimizer logs.

The supplied workbook produces 24,506 training and 8,774 test cases. These are audited outputs, not hard-coded count targets. The analysis concerns retained elective workloads, not complete historical hospital operations.

See \`docs/experiment.md\`, \`docs/theory.md\`, and \`docs/validation.md\`.
