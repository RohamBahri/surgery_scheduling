"""Command-line pilot for fixed-capacity W2b and three decision-aware losses."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from w2b.data import load_weeks
from w2b.learning import fit_case_error, train_method
from w2b.planner import cached_solve, implement, schedule_cost


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("workbook", type=Path)
    p.add_argument("--out", type=Path, default=Path("results/w2b"))
    p.add_argument("--group", choices=("TGH", "TWH-main", "TWH-day"), default="TWH-day")
    p.add_argument("--scenario", nargs=2, type=float, metavar=("ALPHA", "H"),
                   default=(.8, 30.))
    p.add_argument("--method", choices=("vf", "gap", "spo", "all"), default="all")
    p.add_argument("--weeks", type=int, default=2,
                   help="First N training weeks below cap; 0 means all")
    p.add_argument("--max-cases", type=int, default=70,
                   help="Exclude weeks above N cases; 0 means no limit")
    p.add_argument("--max-move-days", type=int, default=4)
    p.add_argument("--move-penalty", type=float, default=0.)
    p.add_argument("--gamma", type=float, default=2.,
                   help="Gamma >= 1 for gap and SPO-style losses")
    p.add_argument("--outer", type=int, default=2)
    p.add_argument("--inner-evals", type=int, default=150)
    p.add_argument("--initial-evals", type=int, default=700)
    p.add_argument("--lambda-l1", type=float, default=1.)
    p.add_argument("--seconds", type=float, default=120.,
                   help="Time limit for each ordinary Gurobi MILP")
    p.add_argument("--adversary-seconds", type=float, default=120.)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--init-weights", type=Path,
                   help="Optional JSON array or daily-experiment models_GROUP.json")
    args = p.parse_args(argv)
    alpha, h = args.scenario
    if not (0 < alpha <= 1 and h > 0 and args.gamma >= 1
            and args.weeks >= 0 and args.max_cases >= 0
            and 0 <= args.max_move_days <= 4
            and args.move_penalty >= 0 and args.outer >= 0
            and args.seconds > 0 and args.adversary_seconds > 0
            and args.threads >= 1 and args.inner_evals >= 1):
        p.error("Invalid response, scale, movement, loss or solver argument")
    if not args.workbook.is_file():
        p.error(f"Workbook not found: {args.workbook}")
    fingerprint = sha256(args.workbook.read_bytes()).hexdigest()
    scientific = dict(group=args.group, alpha=alpha, h=h,
                      weeks=args.weeks, max_cases=args.max_cases,
                      move_penalty=args.move_penalty,
                      max_move_days=args.max_move_days,
                      workbook_sha=fingerprint, gamma=args.gamma,
                      l1=args.lambda_l1,
                      initial_weights=str(args.init_weights) if args.init_weights else "estimated")
    runid = sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest()[:12]
    root = args.out / runid
    root.mkdir(parents=True, exist_ok=True)
    print("W2b output directory:", root.resolve(), flush=True)
    selected, meta = load_weeks(args.workbook, group=args.group, weeks=args.weeks,
                                max_cases=args.max_cases,
                                max_move_days=args.max_move_days)
    (root / "specification.json").write_text(
        json.dumps({"scientific": scientific, "cohort": meta}, indent=2) + "\n")
    cache = root / "cache"

    if args.init_weights:
        obj = json.loads(args.init_weights.read_text())
        if isinstance(obj, list):
            start = np.asarray(obj, float)
        else:
            key = f"{alpha:g}_{h:g}"
            start = np.asarray(obj["scenarios"][key]["Case-Error"]["w"], float)
        init_info = {"source": str(args.init_weights)}
    else:
        print("Fitting response-aware case-error initializer...", flush=True)
        start, init_info = fit_case_error(selected, alpha=alpha, h=h,
                                          max_evals=args.initial_evals)
    if len(start) != selected[0].X.shape[1] or not np.isfinite(start).all():
        raise ValueError("Initializer has wrong feature dimension/nonfinite coefficients")
    (root / "initialization.json").write_text(
        json.dumps({"weights": start.tolist(), **init_info}, indent=2) + "\n")

    methods = ("vf", "gap", "spo") if args.method == "all" else (args.method,)
    summary = []
    for method in methods:
        print(f"\nTraining {method} on {len(selected)} weeks "
              f"({sum(w.n for w in selected)} cases), alpha={alpha}, h={h}, "
              f"gamma={args.gamma}", flush=True)
        path = root / (method + "_training.json")
        result = train_method(
            method, selected, start=start, alpha=alpha, h=h,
            gamma=args.gamma, lam=args.lambda_l1, outer=args.outer,
            inner_evals=args.inner_evals, seconds=args.seconds,
            adversary_seconds=args.adversary_seconds,
            threads=args.threads, cache=cache, move_penalty=args.move_penalty)
        path.write_text(json.dumps(result, indent=2) + "\n")
        last = result["iterations"][-1]["evaluation"]
        summary.append({"method": method, "realized_cost": last["realized"],
                        "regret": last["regret"],
                        "library_proxy": result["iterations"][-1]["library_proxy"],
                        "certified_oracles": last["certified"],
                        "planner_incomplete": last["planner_incomplete"],
                        "adversary_incomplete": last["adversary_incomplete"]})
        print("  realized mean:", last["realized"],
              "regret:", last["regret"],
              "oracle complete:", last["certified"], flush=True)
    (root / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\nDone. These are TRAINING-only, experimental local-search results.")
    print("Neither the restricted-library proxies nor their optimizers give a "
          "global policy-optimality certificate.")


if __name__ == "__main__":
    main()
