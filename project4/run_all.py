"""Regenerate every figure and table:  python -m project4.run_all

Each experiment module in EXPERIMENTS must define main(). Modules that do
not exist yet are reported as skipped, so this works from week 1 onwards.
Options:  --only t6_analytic task2_cost_sweep   (run a subset)
"""
import argparse
import importlib
import time
import traceback

EXPERIMENTS = (
    ("t6_analytic", "M6  T6 analytic curves and rule table"),
    ("c2_gaussian", "M3  C2 Gaussian classifiers on Spambase, raw vs log(1+x)"),
    ("task2_cost_sweep", "M2  C3 logistic regression and cost-ratio sweep"),
    ("task3_calibration", "M5  C4 log-loss, Brier, reliability"),
    ("task4_prior_shift", "M1  C5 prior-shift experiments"),
    ("task5_inspection", "M4  C6 accuracy-vs-cost exhibit and inspection curves"),
)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", help="experiment module names to run")
    args = ap.parse_args(argv)
    todo = [e for e in EXPERIMENTS if not args.only or e[0] in args.only]
    status = []
    for name, desc in todo:
        print(f"== {name}: {desc}")
        try:
            mod = importlib.import_module(f"project4.experiments.{name}")
        except ModuleNotFoundError as e:
            if e.name == f"project4.experiments.{name}":
                print("   skipped (not written yet)")
                status.append((name, "skipped"))
                continue
            raise
        t = time.time()
        try:
            mod.main()
            status.append((name, f"ok ({time.time() - t:.1f}s)"))
        except Exception:
            traceback.print_exc()
            status.append((name, "FAILED"))
    print("\nSummary")
    for name, s in status:
        print(f"  {name:20s} {s}")
    return 1 if any(s == "FAILED" for _, s in status) else 0


if __name__ == "__main__":
    raise SystemExit(main())
