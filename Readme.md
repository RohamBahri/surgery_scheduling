# Predictive & Stochastic Optimization for Elective Surgery Scheduling
A single-script prototype (`surgery_scheduling.py`) that compares four
approaches for scheduling elective surgeries under uncertain case durations:

| Method | Key idea | Solver |
|--------|----------|--------|
| **Deterministic** | Uses booked time as fixed duration | Gurobi MIP |
| **SAA** | Two-stage stochastic program with 30 empirical scenarios | Gurobi MIP |
| **LASSO-Predictive** | Duration predicted by one-off LASSO regression | Gurobi MIP |
| **KNN-Predictive** | \(k\)-NN predictor tuned on a cost proxy (overtime + idle) | Gurobi MIP |

The script runs a rolling-horizon simulation (default: **10 horizons × 7 days**)
and prints aggregated KPIs (planned objective, actual cost, OT/idle minutes, etc.).