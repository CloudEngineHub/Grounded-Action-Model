"""Score a LIBERO-PRO run: success rate per cell (suite x variant) and overall.

    python benchmarks/libero_pro/score.py --config configs/libero_pro.yaml [--out runs/libero_pro]

Each cell is scored over its 10 tasks x the episodes they were supposed to run (min(episodes_per_task,
number of initial states); spatial tasks have 20). Units that are not in eval.task_list, or episodes
that were not run, count as failures, so a partial run gives a lower bound.
"""
import argparse, glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from GAM.config import load_yaml
from eval import SUITES, VARIANTS, suite_dir
from run_all import units_of


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--out", default=None)
    a = ap.parse_args(); cfg = load_yaml(a.config); out = a.out or cfg["eval"]["out_dir"]
    n_eps = int(cfg["eval"]["episodes_per_task"]); run = set(units_of(cfg))
    from eval import task_dirs
    import torch
    rows, T, cell_rates = [], [0, 0, 0], []
    print(f"{'cell':<18s}{'success':>12s}{'rate':>8s}{'run':>7s}  per task (success/run, '-' = not in task_list); rate = mean over the 10 tasks")
    for s in SUITES:
        for v in VARIANTS:
            inits = sorted(glob.glob(os.path.join(task_dirs(cfg, suite_dir(s, v))[1], "*.pruned_init")))
            res = {}
            for ln in (open(os.path.join(out, f"{s}_{v}", "results.jsonl")) if os.path.exists(os.path.join(out, f"{s}_{v}", "results.jsonl")) else []):
                try:
                    r = json.loads(ln)
                    if "success" in r:
                        res[(r["ti"], r["ep"])] = bool(r["success"])
                except Exception:
                    pass
            cs = cn = cr = 0; parts = []; task_rates = []
            for ti in range(10):
                need = min(n_eps, len(torch.load(inits[ti], weights_only=False))) if ti < len(inits) else n_eps
                cn += need
                if (s, v, str(ti)) not in run and (s, v, ti) not in run:
                    parts.append(f"t{ti}:-"); task_rates.append(0.0); continue   # a task that was not run counts as 0
                ok = sum(res.get((ti, e), False) for e in range(need)); n = sum((ti, e) in res for e in range(need))
                cs += ok; cr += n; parts.append(f"t{ti}:{ok}/{n}" + ("" if n >= need else "*")); task_rates.append(ok / need)
            rate = 100 * sum(task_rates) / 10; cell_rates.append(rate)
            T[0] += cs; T[1] += cn; T[2] += cr
            print(f"{s + '/' + v:<18s}{f'{cs}/{cn}':>12s}{rate:7.1f}%{cr:>7d}  " + " ".join(parts))
    print(f"{'mean of 16 cells':<18s}{f'{T[0]}/{T[1]}':>12s}{sum(cell_rates) / len(cell_rates):7.1f}%{T[2]:>7d}   (* = task not fully run; each task weighs 1/10 of its cell regardless of its episode count)")


if __name__ == "__main__":
    main()
