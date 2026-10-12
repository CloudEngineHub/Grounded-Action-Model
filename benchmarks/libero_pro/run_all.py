"""Run the whole LIBERO-PRO benchmark (every unit of eval.task_list) on the GPUs of this machine.

    python benchmarks/libero_pro/run_all.py --config configs/libero_pro.yaml [--gpus 0,1,2,3] [--episodes 50] [--ckpt my.pt]

A unit is one (suite, variant, task). Units are pulled from a shared queue by one worker per GPU;
each worker calls eval.py, so finished episodes are skipped when the run is restarted.
Afterwards `python benchmarks/libero_pro/score.py --config configs/libero_pro.yaml` prints the table.
"""
import argparse, os, queue, subprocess, sys, threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from GAM.config import load_yaml
from eval import SUITES, VARIANTS


def units_of(cfg):
    tl = cfg["eval"].get("task_list", "all")
    if tl in (None, "all"):
        return [(s, v, t) for s in SUITES for v in VARIANTS for t in range(10)]
    return [tuple(l.split()[:3]) for l in open(tl) if l.strip() and not l.startswith("#")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--gpus", default="0")
    ap.add_argument("--episodes", type=int, default=None); ap.add_argument("--ckpt", default=None)
    ap.add_argument("--chunking", default=None); ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    cfg = load_yaml(a.config); units = units_of(cfg)
    by_cell = {}
    for s, v, t in units:
        by_cell.setdefault((s, v), []).append(str(t))
    q = queue.Queue()
    for (s, v), ts in by_cell.items():
        q.put((s, v, ",".join(ts)))
    print(f"[run_all] {len(units)} units in {q.qsize()} cells on GPUs {a.gpus}", flush=True)
    extra = [] + (["--episodes", str(a.episodes)] if a.episodes else []) + (["--ckpt", a.ckpt] if a.ckpt else []) \
        + (["--chunking", a.chunking] if a.chunking else []) + (["--no-video"] if a.no_video else [])

    def worker(gpu):
        while True:
            try:
                s, v, ts = q.get_nowait()
            except queue.Empty:
                return
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu))
            cmd = [sys.executable, os.path.join(HERE, "eval.py"), "--config", a.config, "--suite", s, "--variant", v, "--tasks", ts] + extra
            print(f"[gpu{gpu}] {s}/{v} tasks {ts}", flush=True)
            rc = subprocess.call(cmd, env=env)
            print(f"[gpu{gpu}] {s}/{v} done (exit {rc})", flush=True)

    ths = [threading.Thread(target=worker, args=(g,)) for g in a.gpus.split(",")]
    [t.start() for t in ths]; [t.join() for t in ths]
    print("[run_all] done", flush=True)


if __name__ == "__main__":
    main()
